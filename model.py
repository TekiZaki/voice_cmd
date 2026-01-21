import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
SAMPLE_RATE = 44100
DURATION = 1.0
N_MFCC = 40
EPOCHS = 100
BATCH_SIZE = 16

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def load_data():
    X = []
    y = []
    
    labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    print(f"📂 Detected classes: {labels}")
    
    # Count samples per class
    sample_counts = {}
    
    for label in labels:
        class_path = os.path.join(DATASET_PATH, label)
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        sample_counts[label] = len(wav_files)
        
        print(f"   {label}: {len(wav_files)} samples")
        
        for file in wav_files:
            file_path = os.path.join(class_path, file)
            
            try:
                # Load audio
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # Normalize audio
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                
                # Pad if too short
                if len(audio) < SAMPLE_RATE * DURATION:
                    audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
                else:
                    audio = audio[:int(SAMPLE_RATE * DURATION)]
                
                # Original Sample
                mfcc = extract_mfcc(audio)
                X.append(mfcc)
                y.append(label)
                
                # --- DATA AUGMENTATION (only if we have few samples) ---
                if len(wav_files) < 15:
                    # Add white noise
                    noise_audio = add_noise(audio, noise_factor=0.005)
                    X.append(extract_mfcc(noise_audio))
                    y.append(label)
                    
                    # Time shift
                    shifted_audio = shift_time(audio, shift_max=0.2)
                    X.append(extract_mfcc(shifted_audio))
                    y.append(label)
                    
                    # Pitch shift (slight variation)
                    pitch_audio = pitch_shift(audio)
                    X.append(extract_mfcc(pitch_audio))
                    y.append(label)
                    
            except Exception as e:
                print(f"   ⚠️ Error loading {file}: {e}")
                continue
    
    # Check for insufficient data
    print("\n📊 Sample Analysis:")
    for label, count in sample_counts.items():
        status = "✅" if count >= 10 else "⚠️ TOO FEW!"
        print(f"   {label}: {count} samples {status}")
    
    if any(count < 5 for count in sample_counts.values()):
        print("\n🚨 WARNING: Some classes have fewer than 5 samples!")
        print("   Recommendation: Collect at least 15-20 samples per command.\n")
    
    # Synthetic background noise
    print("\n🔊 Generating synthetic 'background' noise...")
    num_bg_samples = max(30, len(X) // len(labels))  # Proportional to dataset
    for i in range(num_bg_samples):
        bg_noise = np.random.normal(0, 0.003, int(SAMPLE_RATE * DURATION))
        X.append(extract_mfcc(bg_noise))
        y.append("background")
    print(f"   Generated {num_bg_samples} background samples")

    return np.array(X), np.array(y)

def extract_mfcc(audio):
    """Extract MFCC features with delta and delta-delta"""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Add delta and delta-delta for better feature representation
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine all features
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    return combined.T

def add_noise(audio, noise_factor=0.005):
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def shift_time(audio, shift_max=0.2):
    shift = int(SAMPLE_RATE * shift_max * np.random.uniform(-1, 1))
    return np.roll(audio, shift)

def pitch_shift(audio, n_steps=None):
    """Shift pitch slightly"""
    if n_steps is None:
        n_steps = np.random.randint(-2, 3)  # Random pitch shift between -2 and +2 semitones
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)

def build_model(input_shape, num_classes):
    """Improved model architecture"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Reshape for Conv2D
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a good optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 VOICE COMMAND MODEL TRAINER")
    print("=" * 60)
    
    # 1. Load and Preprocess
    print("\n📦 Loading dataset...")
    X, y = load_data()
    
    if len(X) == 0:
        print("\n❌ ERROR: No data loaded! Check your dataset folder.")
        exit(1)
    
    print(f"\n✅ Total samples loaded: {len(X)}")
    
    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"📝 Number of classes: {num_classes}")
    print(f"📝 Classes: {list(le.classes_)}")
    
    # Check class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    print("\n📊 Class Distribution:")
    for cls, count in zip(le.classes_, counts):
        print(f"   {cls}: {count} samples")
    
    # Warning if imbalanced
    if max(counts) / min(counts) > 3:
        print("\n⚠️  WARNING: Classes are imbalanced! Consider collecting more data for underrepresented classes.")
    
    # Save Label Encoder
    np.save(os.path.join(MODELS_PATH, 'label_encoder.npy'), le.classes_)
    print(f"\n💾 Label encoder saved")
    
    # Split Data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\n🔀 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # 2. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n🏗️  Input shape: {input_shape}")
    
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # 4. Train
    print("\n🚀 Training model...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    
    print(f"\n✅ Training Accuracy: {train_acc*100:.2f}%")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    if test_acc < 0.7:
        print("\n  WARNING: Test accuracy is below 70%!")
        print("   Recommendations:")
        print("   1. Collect more training samples (15-20 per command)")
        print("   2. Ensure recordings are clear and consistent")
        print("   3. Reduce the number of commands if you have too many")
        print("   4. Check that recordings are properly labeled")
    
    # 6. Save Model
    model_save_path = os.path.join(MODELS_PATH, 'voice_model.keras')
    model.save(model_save_path)
    print(f"\n💾 Model saved at: {model_save_path}")
    
    # Also save as .h5 for compatibility
    h5_path = os.path.join(MODELS_PATH, 'voice_model.h5')
    model.save(h5_path)
    print(f"💾 Legacy format saved at: {h5_path}")
    
    print("\n" + "=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)