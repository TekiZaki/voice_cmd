import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- KONFIGURASI ---
DATASET_PATH = 'dataset'        # Jalur folder dataset
MODELS_PATH = 'models'          # Jalur folder penyimpanan model
SAMPLE_RATE = 44100             # Tingkat sampling audio
DURATION = 2.0                  # Durasi audio dalam detik
N_MFCC = 40                     # Jumlah koefisien MFCC yang diekstrak
EPOCHS = 70                     # Jumlah iterasi pelatihan
BATCH_SIZE = 32                 # Ukuran batch untuk pelatihan

# Membuat direktori model jika belum ada
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def load_data():
    """Memuat data audio dari dataset dan menerapkan augmentasi."""
    X = []  # List untuk fitur audio
    y = []  # List untuk label kelas
    
    # Mendapatkan daftar label dari nama folder di direktori dataset
    labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('_')]
    
    print(f"📂 Kelas yang terdeteksi: {labels}")
    
    for label in labels:
        class_path = os.path.join(DATASET_PATH, label)
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        print(f"   Memproses {label}: {len(wav_files)} sampel asli...")
        
        for file in wav_files:
            file_path = os.path.join(class_path, file)
            
            try:
                # Memuat file audio dengan tingkat sampling yang ditentukan
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Menyesuaikan durasi audio (tambahkan padding jika kurang, potong jika lebih)
                target_samples = int(SAMPLE_RATE * DURATION)
                if len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                else:
                    audio = audio[:target_samples]
                
                # Normalisasi amplitudo audio
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                
                # Ekstraksi fitur MFCC dari audio asli
                X.append(extract_mfcc(audio))
                y.append(label)
                
                # AUGMENTASI DATA
                
                # 1. Menambahkan Noise/Derau (Simulasi lingkungan bising)
                noise_audio = add_noise(audio, noise_factor=0.008)
                X.append(extract_mfcc(noise_audio))
                y.append(label)
                
                # 2. Pergeseran Waktu (Geser audio secara horizontal)
                shifted_audio = shift_time(audio, shift_max=0.2)
                X.append(extract_mfcc(shifted_audio))
                y.append(label)
                
                # 3. Pergeseran Nada (Ubah nada audio secara acak)
                step = np.random.randint(-2, 3)
                if step != 0:
                    pitch_audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=step)
                    X.append(extract_mfcc(pitch_audio))
                    y.append(label)

            except Exception as e:
                print(f"Gagal memproses {file}: {e}")
                continue

    # PENANGANAN NOISE LATAR BELAKANG (BACKGROUND NOISE)
    bg_path = os.path.join(DATASET_PATH, '_background_noise')
    bg_samples = []
    
    # Jika folder noise latar tersedia, muat sampelnya
    if os.path.exists(bg_path):
        print("\nMemuat sampel noise latar belakang asli...")
        for file in os.listdir(bg_path):
            if file.endswith('.wav'):
                try:
                    fp = os.path.join(bg_path, file)
                    aud, _ = librosa.load(fp, sr=SAMPLE_RATE)
                    target = int(SAMPLE_RATE * DURATION)
                    # Potong noise menjadi potongan-potongan sesuai durasi target
                    for i in range(0, len(aud) - target, target):
                        chunk = aud[i:i+target]
                        X.append(extract_mfcc(chunk))
                        y.append("background")
                        bg_samples.append(chunk)
                except: pass
    
    # Jika sampel noise kurang dari 50, buat noise putih secara sintetik
    if len(bg_samples) < 50:
        print("Menghasilkan noise latar belakang sintetik...")
        num_needed = 100 - len(bg_samples)
        for _ in range(num_needed):
            noise = np.random.normal(0, 0.005, int(SAMPLE_RATE * DURATION))
            X.append(extract_mfcc(noise))
            y.append("background")

    return np.array(X), np.array(y)

def extract_mfcc(audio):
    """Mengekstrak fitur MFCC beserta delta dan delta-delta."""
    # Ekstraksi MFCC standar
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Menghitung delta (turunan pertama)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Menghitung delta2 (turunan kedua)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Menggabungkan ketiga fitur menjadi satu array
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    # Transpose agar dimensi sesuai dengan input CNN (Frames, Fitur)
    return combined.T

def add_noise(audio, noise_factor):
    """Menambahkan noise putih ke dalam audio."""
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def shift_time(audio, shift_max):
    """Menggeser audio dalam domain waktu secara acak."""
    shift = int(SAMPLE_RATE * shift_max * np.random.uniform(-1, 1))
    return np.roll(audio, shift)

def build_compact_model(input_shape, num_classes):
    """Membangun arsitektur model CNN yang ringan."""
    model = models.Sequential([
        layers.Input(shape=input_shape), # Layer input
        
        # Reshape untuk input Conv2D (menambah dimensi channel satu)
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # Blok Konvolusi 1
        layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Blok Konvolusi 2
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Blok Konvolusi 3
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(), # Mengubah output 2D menjadi 1D
        
        # Fully Connected Layer (Dense)
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer (Klasifikasi Softmax)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Menggunakan optimizer Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Kompilasi model dengan loss function yang sesuai
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("TRAINING MODEL PERINTAH SUARA")
    print("="*60)
    
    # 1. Memuat Dataset
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: Dataset kosong.")
        exit()
        
    print(f"\nTotal Sampel Pelatihan (Setelah Augmentasi): {len(X)}")
    
    # 2. Encoding Label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    # Menyimpan daftar label kelas
    np.save(os.path.join(MODELS_PATH, 'label_encoder.npy'), le.classes_)
    
    print(f"Kelas yang dipelajari ({num_classes}): {le.classes_}")

    # 3. Pembagian Data (Training & Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Pembangunan & Pelatihan Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\nBentuk Input Model: {input_shape}")
    
    model = build_compact_model(input_shape, num_classes)
    model.summary() # Menampilkan ringkasan arsitektur model
    
    # Definisi Callback untuk optimasi pelatihan
    callbacks = [
        # Berhenti lebih awal jika tidak ada peningkatan
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        # Mengurangi learning rate saat stagnan
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]
    
    print("\nMemulai Pelatihan...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Penyimpanan Model
    model.save(os.path.join(MODELS_PATH, 'voice_model.h5'))
    model.save(os.path.join(MODELS_PATH, 'voice_model.keras'))
    
    print("\nPelatihan Selesai & Model Berhasil Disimpan!")
    
    # Evaluasi akhir menggunakan data testing
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Akurasi Pengujian Akhir: {acc*100:.2f}%")