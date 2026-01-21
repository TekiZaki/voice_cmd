import numpy as np
import librosa
import noisereduce as nr
# import scipy if needed later

def enhance_audio(audio, sample_rate=44100):
    """
    Apply a suite of enhancements to improve recognition accuracy.
    1. Denoising (Noise Reduction)
    2. Pre-emphasis (High-frequency boost)
    3. Normalization (Loudness leveling)
    """
    if len(audio) == 0:
        return audio
        
    # --- 1. Noise Reduction ---
    # We use stationary noise reduction. For live audio, we assume the first few 
    # ms or the overall low-energy parts are noise.
    try:
        # If audio is long enough, use the first 2000 samples as noise profile
        # Otherwise use the whole clip if it's very short (not recommended but safe)
        noise_clip = audio[:2000] if len(audio) > 2000 else audio
        audio_denoised = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noise_clip, prop_decrease=0.8)
    except Exception as e:
        print(f"⚠️ Denoising failed: {e}")
        audio_denoised = audio

    # --- 2. Pre-emphasis ---
    # Formula: y(t) = x(t) - 0.97 * x(t-1)
    # This boosts higher frequencies which are critical for speech features (MFCCs)
    pre_emphasis = 0.97
    audio_emphasized = np.append(audio_denoised[0], audio_denoised[1:] - pre_emphasis * audio_denoised[:-1])

    # --- 3. Normalization (Peak) ---
    # Ensure the audio uses the full dynamic range without clipping
    max_val = np.max(np.abs(audio_emphasized))
    if max_val > 0:
        audio_normalized = audio_emphasized / max_val
    else:
        audio_normalized = audio_emphasized

    # --- 4. Trimming ---
    # Remove leading/trailing silence to focus on the command
    audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)
    
    return audio_trimmed
