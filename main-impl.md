# Plan: Implement Voice Assistant (main.py)

## Overview

Implement `main.py` to act as the real-time Voice Assistant. It will listen to audio from the microphone, transform it into MFCC features, use the trained model for classification, and execute the mapped application shortcuts.

## Project Type

BACKEND / DESKTOP (Deep Learning / Windows)

## Success Criteria

- [x] Successfully loads `models/voice_model.h5` and `models/label_encoder.npy`.
- [x] Successfully loads `command_map.json`.
- [x] Listens to microphone input in real-time.
- [x] Correctly identifies commands (with a confidence threshold).
- [x] Opens the correct `.lnk` file in `apps/` using `os.startfile`.

## Tech Stack

- Python 3.x
- TensorFlow / Keras (Model Inference)
- Librosa (Audio Preprocessing)
- Sounddevice / Scipy (Real-time Recording)
- OS / Subprocess (Executing Shortcuts)

## Task Breakdown

### Phase 1: Preparation

- [x] Load model, label encoder, and command map.
- [x] Set up global audio configurations (matching `model.py`).

### Phase 2: Audio Capture & Processing

- [x] Implement audio stream listener using `sounddevice`.
- [x] Implement VAD (Voice Activity Detection) or simple RMS thresholding.
- [x] Implement audio-to-MFCC transformation for inference.

### Phase 3: Inference & Execution

- [x] Pass MFCC to the model for prediction.
- [x] Filter out "background" and low-confidence results.
- [x] Execute `os.startfile()` based on the predicted label.

## Phase X: Verification

- [ ] Run `python main.py` and verify it detects silence correctly.
- [ ] Speak a command and verify the corresponding app opens.
