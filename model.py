import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
SAMPLE_RATE = 44100
DURATION = 1.0  # seconds
N_MFCC = 40
EPOCHS = 50
BATCH_SIZE = 8

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def load_data():
    X = []
    y = []
    
    # Get labels from folder names
    labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    print(f"Detected classes: {labels}")
    
    # Process each class
    for label in labels:
        class_path = os.path.join(DATASET_PATH, label)
        for file in os.listdir(class_path):
            if file.endswith('.wav'):
                file_path = os.path.join(class_path, file)
                
                # Load audio
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # Pad if too short
                if len(audio) < SAMPLE_RATE * DURATION:
                    audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
                
                # Original Sample
                mfcc = extract_mfcc(audio)
                X.append(mfcc)
                y.append(label)
                
                # --- DATA AUGMENTATION ---
                # 1. Add Noise
                noise_audio = add_noise(audio)
                X.append(extract_mfcc(noise_audio))
                y.append(label)
                
                # 2. Time Shifting
                shifted_audio = shift_time(audio)
                X.append(extract_mfcc(shifted_audio))
                y.append(label)

    # --- SYNTHETIC BACKGROUND NOISE ---
    print("Generating synthetic 'background' noise class...")
    for i in range(20): # Generate 20 samples of background
        # Generated from very low amplitude random noise (silence simulation)
        bg_noise = np.random.normal(0, 0.005, int(SAMPLE_RATE * DURATION))
        X.append(extract_mfcc(bg_noise))
        y.append("background")

    return np.array(X), np.array(y)

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # Add channel dimension
    return mfcc.T

def add_noise(audio):
    noise = np.random.normal(0, 0.01, len(audio))
    return audio + noise

def shift_time(audio):
    shift = int(SAMPLE_RATE * 0.1) # 100ms shift
    return np.roll(audio, shift)

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Reshape for Conv2D (Height, Width, Channel)
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("--- Starting Voice Model Training ---")
    
    # 1. Load and Preprocess
    print("Loading dataset...")
    X, y = load_data()
    
    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Save Label Encoder
    np.save(os.path.join(MODELS_PATH, 'label_encoder.npy'), le.classes_)
    print(f"Label encoder saved. Classes: {le.classes_}")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 2. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # 3. Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # 4. Save Model
    model_save_path = os.path.join(MODELS_PATH, 'voice_model.h5')
    model.save(model_save_path)
    print(f"\nModel saved successfully at: {model_save_path}")
    
    # 5. Quick Eval
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
