# Plan: Implement Voice Command Training Model

## Overview

Implement `model.py` to train a CNN model for voice command recognition. The script will handle data loading, feature extraction (MFCC), data augmentation, synthetic noise generation, and model training/saving.

## Project Type

BACKEND (Deep Learning / Python)

## Success Criteria

- [x] `model.py` successfully reads from `dataset/` folder.
- [x] Synthetic "Silence/Background" class is generated.
- [x] Data augmentation (Noise, Shift) is implemented.
- [x] CNN model is trained and saved as `models/voice_model.h5`.
- [x] `label_encoder.npy` is saved for prediction mapping.

## Tech Stack

- Python 3.x
- TensorFlow / Keras (CNN)
- Librosa (Audio processing)
- NumPy (Data manipulation)

## File Structure

- `model.py`: Main training script.
- `models/`: Folder to store weights and label encoder.
- `dataset/`: Training data source.

## Task Breakdown

### Phase 1: Analysis & Preparation

- [x] Verify `dataset` structure and file contents.
- [x] Ensure `models/` directory exists.

### Phase 2: Feature Extraction & Preprocessing

- [x] Implement `extract_features(file_path)` using Librosa.
- [x] Implement `data_augmentation(audio)` (Noise, Time Shifting).
- [x] Implement `generate_synthetic_noise()` to create the 'background' class.

### Phase 3: Model Architecture

- [x] Define CNN architecture (Conv2D -> MaxPooling -> Dropout -> Flatten -> Dense).
- [x] Compile model with `adam` optimizer and `sparse_categorical_crossentropy`.

### Phase 4: Training & Export

- [x] Train the model with early stopping.
- [x] Save `voice_model.h5` and `label_encoder.npy`.

## Phase X: Verification

- [x] Run `python model.py` and verify it completes without errors.
- [x] Check if files appear in `models/`.
