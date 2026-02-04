import numpy as np
import librosa
import noisereduce as nr

def enhance_audio(audio, sample_rate=44100):
    """
    Menerapkan serangkaian perbaikan untuk meningkatkan akurasi pengenalan suara.
    Mencakup: Reduksi noise, Pre-emphasis, Normalisasi, dan Pemotongan (Trimming).
    """
    # Mengembalikan audio jika kosong
    if len(audio) == 0:
        return audio
        
    # --- 1. Reduksi Noise (Denoising) ---
    try:
        # Gunakan 2000 sampel pertama sebagai profil noise jika memungkinkan
        noise_clip = audio[:2000] if len(audio) > 2000 else audio
        # Mengurangi noise statis pada sinyal audio
        audio_denoised = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noise_clip, prop_decrease=0.8)
    except Exception as e:
        # Jika gagal, gunakan audio asli tanpa reduksi noise
        print(f"Reduksi noise gagal: {e}")
        audio_denoised = audio

    # --- 2. Pre-emphasis ---
    # Memperkuat frekuensi tinggi yang penting untuk fitur bicara (MFCC)
    pre_emphasis = 0.97
    # Rumus: y(t) = x(t) - 0.97 * x(t-1)
    audio_emphasized = np.append(audio_denoised[0], audio_denoised[1:] - pre_emphasis * audio_denoised[:-1])

    # --- 3. Normalisasi Amplitudo ---
    # Menghitung nilai puncak maksimum
    max_val = np.max(np.abs(audio_emphasized))
    # Skalakan audio ke rentang dinamis penuh agar volume seragam
    if max_val > 0:
        audio_normalized = audio_emphasized / max_val
    else:
        audio_normalized = audio_emphasized

    # --- 4. Trimming (Pemotongan Sunyi) ---
    # Menghapus bagian sunyi di awal dan akhir audio agar fokus pada perintah
    audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)
    
    # Mengembalikan audio yang telah diproses
    return audio_trimmed
