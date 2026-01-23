import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
# Parameter Audio
SAMPLE_RATE = 44100
DURATION = 2.0
N_MFCC = 40
# Parameter Training
EPOCHS = 70          # Dikurangi sedikit karena model lebih efisien
BATCH_SIZE = 32      # Dinaikkan agar gradient lebih stabil

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def load_data():
    X = []
    y = []
    
    # Ambil list folder, abaikan folder hidden
    labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('_')]
    
    print(f"📂 Detected classes: {labels}")
    
    for label in labels:
        class_path = os.path.join(DATASET_PATH, label)
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        print(f"   Processing {label}: {len(wav_files)} raw samples...")
        
        for file in wav_files:
            file_path = os.path.join(class_path, file)
            
            try:
                # 1. Load Audio
                # Note: Kita TIDAK melakukan enhance_audio lagi disini karena 
                # data_collector.py sudah melakukannya. Double process = suara rusak.
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # 2. Fix Duration (Pad or Trim)
                target_samples = int(SAMPLE_RATE * DURATION)
                if len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                else:
                    audio = audio[:target_samples]
                
                # 3. Normalize (Penting untuk neural net)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                
                # --- DATASET ASLI ---
                X.append(extract_mfcc(audio))
                y.append(label)
                
                # --- AUGMENTASI (DITERAPKAN KE SEMUA DATA) ---
                # Ini kunci agar model pintar di dunia nyata, bukan cuma saat training.
                
                # Var 1: Noise Injection (Simulasi ruangan berisik)
                noise_audio = add_noise(audio, noise_factor=0.008)
                X.append(extract_mfcc(noise_audio))
                y.append(label)
                
                # Var 2: Time Shift (Geser waktu bicara sedikit)
                shifted_audio = shift_time(audio, shift_max=0.2)
                X.append(extract_mfcc(shifted_audio))
                y.append(label)
                
                # Var 3: Pitch Shift (Variasi intonasi)
                # Random shift antara -2 sampai +2 semitone
                step = np.random.randint(-2, 3)
                if step != 0:
                    pitch_audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=step)
                    X.append(extract_mfcc(pitch_audio))
                    y.append(label)

            except Exception as e:
                print(f"   ⚠️ Error processing {file}: {e}")
                continue

    # --- HANDLING BACKGROUND NOISE ---
    # Jika ada folder '_background_noise', pakai itu. Jika tidak, generate sintetik.
    bg_path = os.path.join(DATASET_PATH, '_background_noise')
    bg_samples = []
    
    if os.path.exists(bg_path):
        print("\n🔊 Loading REAL background noise samples...")
        for file in os.listdir(bg_path):
            if file.endswith('.wav'):
                try:
                    fp = os.path.join(bg_path, file)
                    aud, _ = librosa.load(fp, sr=SAMPLE_RATE)
                    # Potong-potong menjadi chunk 1 detik
                    target = int(SAMPLE_RATE * DURATION)
                    for i in range(0, len(aud) - target, target):
                        chunk = aud[i:i+target]
                        X.append(extract_mfcc(chunk))
                        y.append("background")
                        bg_samples.append(chunk)
                except: pass
    
    # Jika sampel background asli kurang dari 50, tambah sintetik
    if len(bg_samples) < 50:
        print("🔊 Generating SYNTHETIC background noise (fallback)...")
        num_needed = 100 - len(bg_samples)
        for _ in range(num_needed):
            # Noise putih halus
            noise = np.random.normal(0, 0.005, int(SAMPLE_RATE * DURATION))
            X.append(extract_mfcc(noise))
            y.append("background")

    return np.array(X), np.array(y)

def extract_mfcc(audio):
    """Extract MFCC features with delta and delta-delta"""
    # MFCC standar
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Delta (Kecepatan perubahan)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Delta2 (Percepatan perubahan)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Gabung jadi (N_MFCC * 3, Time_Steps)
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    # Transpose agar bentuknya (Time_Steps, Features) untuk masuk ke CNN/LSTM
    return combined.T

# --- AUGMENTATION UTILS ---
def add_noise(audio, noise_factor):
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def shift_time(audio, shift_max):
    shift = int(SAMPLE_RATE * shift_max * np.random.uniform(-1, 1))
    return np.roll(audio, shift)

def build_compact_model(input_shape, num_classes):
    """
    Model CNN yang lebih 'Langsing' (Lightweight).
    Mengurangi jumlah parameter drastis untuk mencegah Overfitting pada dataset kecil.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Reshape agar bisa masuk Conv2D (menambah dimensi channel)
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # Block 1 (Reduced filters: 32 -> 16)
        layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), # Dropout ringan
        
        # Block 2 (Reduced filters: 64 -> 32)
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3), # Dropout sedang
        
        # Block 3 (Reduced filters: 128 -> 64)
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Flatten & Dense
        layers.Flatten(),
        
        # Dense Layer (Reduced neurons: 128 -> 64)
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Dropout agresif sebelum output
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("🎤 VOICE COMMAND MODEL TRAINER (V2 - ROBUST)")
    print("="*60)
    
    # 1. Load Data
    X, y = load_data()
    
    if len(X) == 0:
        print("❌ Error: Dataset kosong.")
        exit()
        
    print(f"\n✅ Total Training Samples (Augmented): {len(X)}")
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    np.save(os.path.join(MODELS_PATH, 'label_encoder.npy'), le.classes_)
    
    print(f"📝 Classes ({num_classes}): {le.classes_}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Build & Train
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n🏗️  Model Input Shape: {input_shape}")
    
    model = build_compact_model(input_shape, num_classes)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]
    
    print("\n🚀 Starting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Save
    model.save(os.path.join(MODELS_PATH, 'voice_model.h5'))
    model.save(os.path.join(MODELS_PATH, 'voice_model.keras'))
    
    print("\n✅ Training Complete & Model Saved!")
    
    # Final check
    loss, acc = model.evaluate(X_test, y_test)
    print(f"📊 Final Test Accuracy: {acc*100:.2f}%")