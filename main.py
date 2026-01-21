import os
import json
import time
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sys
import pyautogui
import ctypes
import threading
from audio_utils import enhance_audio

# Set pyautogui safety settings
pyautogui.FAILSAFE = True

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
COMMAND_MAP_PATH = os.path.join(BASE_DIR, 'command_map.json')
APPS_PATH = os.path.join(BASE_DIR, 'apps')
SOUNDS_PATH = os.path.join(BASE_DIR, 'sound')

# Audio Specs (Must match model.py)
SAMPLE_RATE = 44100
DURATION = 1.0  # seconds
N_MFCC = 40

# Detection Settings
RMS_THRESHOLD = 0.05       # Trigger threshold (increase if too sensitive)
CONFIDENCE_THRESHOLD = 0.8 # Minimum probability to execute
COOLDOWN_PERIOD = 1.5      # Seconds to wait after execution

def play_feedback(filename):
    """Play MP3 feedback sound using Windows MCI (non-blocking)."""
    def _play():
        path = os.path.join(SOUNDS_PATH, filename)
        if not os.path.exists(path):
            return
        
        # Unique alias for this play instance
        alias = f"sound_{int(time.time() * 1000)}"
        try:
            # Use short path name to avoid issues with spaces in file path
            ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
        except:
            pass

    threading.Thread(target=_play, daemon=True).start()

def load_resources():
    print("📦 Loading AI models and configurations...")
    try:
        model_path = os.path.join(MODELS_PATH, 'voice_model.h5')
        le_path = os.path.join(MODELS_PATH, 'label_encoder.npy')
        
        if not os.path.exists(model_path) or not os.path.exists(le_path):
            print(f"❌ Error: Model files not found in {MODELS_PATH}")
            print("Please run model.py first.")
            sys.exit(1)
            
        # Load TensorFlow model
        model = tf.keras.models.load_model(model_path)
        
        # Load and reconstruct Label Encoder
        le = LabelEncoder()
        le.classes_ = np.load(le_path, allow_pickle=True)
        
        # Load command to shortcut mapping
        if not os.path.exists(COMMAND_MAP_PATH):
            print(f"❌ Error: {COMMAND_MAP_PATH} not found.")
            sys.exit(1)
            
        with open(COMMAND_MAP_PATH, 'r') as f:
            command_map = json.load(f)
            
        return model, le, command_map
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        sys.exit(1)

def extract_features(audio):
    """Transform raw audio to MFCC features - MUST MATCH model.py!"""
    # --- ENHANCEMENT ---
    audio = enhance_audio(audio, SAMPLE_RATE)

    # Ensure audio is exactly DURATION long
    target_samples = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]
    
    # Normalize audio (safety, enhance_audio already normalizes)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Extract MFCC (same as model.py)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Add delta and delta-delta features (CRITICAL - must match training!)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine all features
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    return combined.T

def execute_action(label, command_map):
    """Opens the mapped application shortcut or performs keyboard action."""
    if label in command_map:
        action = command_map[label]
        
        # Check for keyboard shortcut prefix
        if action.startswith("key:"):
            keys_str = action[4:].strip() # Remove 'key:' prefix
            keys = keys_str.split('+')    # Split 'ctrl+a' into ['ctrl', 'a']
            
            print(f"⌨️ [AUTO] Keyboard Shortcut: {keys_str}")
            try:
                pyautogui.hotkey(*keys)
            except Exception as e:
                print(f"❌ Failed to execute keyboard shortcut: {e}")
            return

        # Default behavior: Open application shortcut
        app_name = action
        app_path = os.path.join(APPS_PATH, app_name)
        
        if os.path.exists(app_path):
            print(f"🚀 [AUTO] Opening application: {app_name}")
            try:
                os.startfile(app_path)
            except Exception as e:
                print(f"❌ Failed to open shortcut: {e}")
        else:
            print(f"⚠️ Shortcut file not found: {app_path}")
    else:
        print(f"ℹ️ Command '{label}' recognized but has no mapping.")

def main():
    # 1. Initialize
    model, le, command_map = load_resources()
    
    print(f"✅ Ready! Detected Commands: {list(le.classes_)}")
    print("\n" + "═"*50)
    print(" 🎤 VOICE ASSISTANT IS LISTENING...")
    print(" 🔊 Threshold: {:.2f} | Confidence Req: {:.1f}%".format(RMS_THRESHOLD, CONFIDENCE_THRESHOLD*100))
    print(" Press Ctrl+C to exit.")
    print("═"*50 + "\n")

    # 2. Setup audio buffer
    buffer_samples = int(DURATION * SAMPLE_RATE)
    audio_buffer = np.zeros(buffer_samples)
    last_action_time = 0
    is_awake = False  # Start in STANDBY mode
    
    # 3. Startup Sound
    play_feedback('standby.mp3')

    def audio_callback(indata, frames, time_info, status):
        """Streaming callback to fill the rolling buffer."""
        nonlocal audio_buffer
        if status:
            print(f"Audio Status: {status}")
        # Shift buffer and add new data
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata.flatten()
    
    # 4. Prediction loop
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            while True:
                current_time = time.time()
                
                # Check cooldown
                if current_time - last_action_time < COOLDOWN_PERIOD:
                    time.sleep(0.1)
                    continue

                # Analyze the last 150ms for voice activity
                check_len = int(0.15 * SAMPLE_RATE)
                rms = np.sqrt(np.mean(audio_buffer[-check_len:]**2))
                
                if rms > RMS_THRESHOLD:
                    # In standby, we can be a bit more quiet about "Analyzing..."
                    if is_awake:
                        print(f"✨ Sound Detected (RMS: {rms:.3f}) - Analyzing...")
                    
                    # Wait briefly for the word to complete in the buffer
                    time.sleep(0.6) 
                    
                    # Snapshot the buffer for inference
                    snapshot = audio_buffer.copy()
                    
                    try:
                        # Extract MFCC & Predict
                        features = extract_features(snapshot)
                        input_data = features[np.newaxis, ...] # Add batch dim (1, Time, Features)
                        
                        predictions = model.predict(input_data, verbose=0)
                        top_idx = np.argmax(predictions[0])
                        confidence = predictions[0][top_idx]
                        label = le.classes_[top_idx]
                        
                        if label == "background":
                            # Ignore background noise
                            pass
                        elif confidence < CONFIDENCE_THRESHOLD:
                            if is_awake:
                                print(f"❓ Low confidence: {label} ({confidence*100:.1f}%)")
                        else:
                            # --- STATE MACHINE LOGIC ---
                            if not is_awake:
                                # In STANDBY: Only respond to wake word
                                if label == "hello_voicecmd":
                                    is_awake = True
                                    print("\n" + "═"*50)
                                    print(" 💡 AWAKE: System is active and ready for commands!")
                                    print("═"*50 + "\n")
                                    play_feedback('active.mp3')
                                    last_action_time = time.time()
                            else:
                                # In AWAKE: Respond to all commands + sleep command
                                if label == "sleep_cmd":
                                    is_awake = False
                                    print("\n" + "═"*50)
                                    print(" 😴 STANDBY: System is sleeping. Say 'Hello VoiceCmD' to wake.")
                                    print("═"*50 + "\n")
                                    play_feedback('standby.mp3')
                                    last_action_time = time.time()
                                else:
                                    print(f"🎯 MATCH: {label.upper()} ({confidence*100:.1f}%)")
                                    execute_action(label, command_map)
                                    last_action_time = time.time()
                    
                    except Exception as e:
                        print(f"❌ Prediction Error: {e}")
                        import traceback
                        traceback.print_exc()
                
                time.sleep(0.05) # Loop breathing room
                
    except KeyboardInterrupt:
        print("\n\n👋 Stopping Voice Assistant. Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()