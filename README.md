# Voice Command Assistant

Sistem asisten kontrol komputer berbasis suara menggunakan jaringan saraf tiruan. Proyek ini mengimplementasikan pendekatan deep learning dengan arsitektur CNN untuk mengenali perintah suara dan mengeksekusi aksi pada sistem operasi Windows.

## Inti dan Tujuan

Voice Command Assistant menyelesaikan masalah interaksi komputer tanpa menggunakan perangkat input fisik seperti keyboard atau mouse. Sistem ini berbeda dari solusi sejenis dengan menggunakan pendekatan kustom yang berfokus pada akurasi pengenalan dalam bahasa Indonesia, integrasi langsung dengan sistem operasi Windows melalui keyboard shortcuts, dan antarmuka pemantauan real-time. Sistem ini dirancang khusus untuk pengguna dengan keterbatasan fisik atau untuk penggunaan hands-free.

Sistem menggunakan teknik ekstraksi fitur MFCC (Mel-Frequency Cepstral Coefficients) dengan delta dan delta-delta untuk representasi audio yang komprehensif. Model dilatih menggunakan arsitektur Convolutional Neural Network dengan tiga blok konvolusi yang masing-masing dilengkapi BatchNormalization dan Dropout untuk mencegah overfitting.

## Fitur Utama

- Pengenalan Perintah Suara Real-Time
  Menggunakan TensorFlow Keras untuk inferensi cepat dengan confidence threshold yang dapat dikonfigurasi. Sistem mendeteksi perintah suara secara terus-menerus dengan memori audio berdurasi 1 detik dan melakukan prediksi ketika level RMS audio melebihi ambang batas yang ditentukan.

- Wake Word dan Sleep Mode
  Memiliki mode aktif dan standby untuk menghemat penggunaan komputasi. Perintah "hello_voicecmd" mengaktifkan sistem, sedangkan "sleep_cmd" menonaktifkannya. Setiap perubahan mode disertai feedback audio yang disesuaikan dalam folder sound/.

- Pemetaan Perintah yang Fleksibel
  Menggunakan file JSON (command_map.json) untuk memetakan label suara ke aksi sistem. Mendukung dua jenis aksi: keyboard shortcuts (contoh: "key:ctrl+c") dan peluncuran aplikasi melalui file shortcut yang disimpan dalam folder apps/.

- Visualisasi dan Logging Real-Time
  Menampilkan antarmuka GUI dengan style HUD yang menunjukkan status sistem, waveform audio secara langsung, riwayat pengenalan dengan confidence score, dan log telemetri untuk debugging dan monitoring performa.

- Peningkatan Kualitas Audio
  Mengimplementasikan pipeline pemrosesan audio yang mencakup noise reduction menggunakan library noisereduce, pre-emphasis untuk meningkatkan frekuensi tinggi, normalisasi peak, dan trimming untuk menghilangkan keheningan di awal dan akhir.

- Augmentasi Data Otomatis
  Saat jumlah sampel training kurang dari 15 per kelas, sistem secara otomatis menambah variasi data melalui penambahan noise white, time shifting, dan pitch shifting untuk meningkatkan robustness model.

## Memulai dengan Cepat

### Prasyarat

- Python 3.8 atau versi lebih baru
- TensorFlow 2.x (terinstal melalui requirements.txt)
- Library audio: sounddevice, librosa, noisereduce
- OS Windows 10/11 untuk integrasi keyboard shortcut dan peluncuran aplikasi
- Mikrofon yang berfungsi dengan baik

### Instalasi

Buka terminal atau command prompt di direktori proyek dan jalankan perintah berikut:

```bash
pip install -r requirements.txt
```

Pastikan direktori berikut telah dibuat secara otomatis oleh aplikasi:
- dataset/ - untuk menyimpan rekaman audio per kategori perintah
- models/ - untuk menyimpan model yang telah dilatih
- apps/ - untuk menyimpan shortcut aplikasi
- sound/ - untuk menyimpan file feedback audio

### Penggunaan Dasar

#### 1. Pengumpulan Data

Jalankan data_collector.py untuk merekam sampel suara:

```bash
python data_collector.py
```

Pada antarmuka GUI, ikuti langkah-langkah berikut:
- Masukkan nama perintah (contoh: "buka_wa", "copy", "paste")
- Klik tombol "Mulai Merekam" untuk merekam sampel suara
- Ucapkan perintah dengan jelas dan konsisten
- Klik "Berhenti Merekam" untuk menyimpan sampel
- Ulangi proses minimal 15-20 kali per perintah untuk akurasi optimal

#### 2. Pelatihan Model

Setelah mengumpulkan data yang cukup, latih model dengan menjalankan:

```bash
python model.py
```

Proses ini akan:
- Memuat semua file audio dari folder dataset/
- Ekstrak fitur MFCC dengan delta dan delta-delta
- Membangun arsitektur CNN 3-layer
- Melakukan training dengan early stopping
- Menyimpan model dalam dua format: voice_model.h5 dan voice_model.keras
- Menyimpan label encoder di models/label_encoder.npy

#### 3. Menjalankan Asisten

Jalankan aplikasi utama:

```bash
python main.py
```

Sistem akan memuat model dan memulai loop inferensi. Ucapkan "hello_voicecmd" untuk mengaktifkan sistem, kemudian ucapkan perintah lain yang telah dilatih. Antarmuka akan menampilkan status aktif/standby, waveform audio real-time, dan riwayat pengenalan.

Contoh kode sederhana untuk melakukan prediksi manual:

```python
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model dan encoder
model = load_model('models/voice_model.h5')
le = LabelEncoder()
le.classes_ = np.load('models/label_encoder.npy', allow_pickle=True)

# Ekstrak fitur MFCC
audio, sr = librosa.load('sample.wav', sr=44100)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0).T

# Prediksi
predictions = model.predict(features[np.newaxis, ...])
predicted_label = le.classes_[np.argmax(predictions)]
confidence = np.max(predictions)
```

## Struktur Proyek

```
voice_cmd/
├── audio_utils.py         - Fungsi pemrosesan audio (denoising, pre-emphasis, normalization)
├── model.py               - Script pelatihan model CNN
├── main.py                - Aplikasi utama dengan GUI dan loop inferensi
├── data_collector.py      - GUI untuk pengumpulan data training
├── command_map.json       - Konfigurasi pemetaan perintah suara ke aksi
├── requirements.txt       - Daftar dependensi Python
├── dataset/               - Direktori sampel audio per kategori
│   ├── hello_voicecmd/    - Sampel perintah wake word
│   ├── sleep_cmd/         - Sampel perintah sleep mode
│   ├── buka_wa/           - Sampel perintah buka WhatsApp
│   └── ...                - Kategori perintah lainnya
├── models/                - Direktori penyimpanan model yang telah dilatih
│   ├── voice_model.h5     - Model dalam format HDF5
│   ├── voice_model.keras  - Model dalam format Keras native
│   └── label_encoder.npy  - Encoder untuk label kelas
├── apps/                  - Shortcut aplikasi untuk peluncuran otomatis
│   ├── WhatsApp Web.lnk   - Shortcut WhatsApp
│   ├── Microsoft Edge.lnk - Shortcut browser
│   └── note.exe - Shortcut.lnk
└── sound/                 - File feedback audio
    ├── active.mp3         - Audio saat sistem aktif
    └── standby.mp3        - Audio saat sistem standby
```

## Konfigurasi

### command_map.json

File konfigurasi utama untuk memetakan label suara ke aksi. Setiap entry terdiri dari pasangan key-value:

```json
{
    "buka_wa": "WhatsApp Web.lnk",
    "copy": "key:ctrl+c",
    "paste": "key:ctrl+v",
    "close": "key:ctrl+w",
    "hello_voicecmd": "none",
    "sleep_cmd": "none"
}
```

Format pemetaan:
- Untuk aplikasi: `"label": "nama_file.lnk"` - File shortcut harus ada di folder apps/
- Untuk keyboard shortcut: `"label": "key:modifier+key"` - Contoh: "key:ctrl+alt+delete"
- Untuk tanpa aksi: `"label": "none"`

### Parameter Utama di main.py

Variabel konfigurasi yang dapat disesuaikan:

- `SAMPLE_RATE = 44100` - Frekuensi sampling audio dalam Hz
- `DURATION = 1.0` - Durasi audio yang diproses per prediksi dalam detik
- `N_MFCC = 40` - Jumlah koefisien MFCC yang diekstrak
- `RMS_THRESHOLD = 0.05` - Ambang batas level audio untuk memicu deteksi
- `CONFIDENCE_THRESHOLD = 0.8` - Ambang batas confidence untuk menerima prediksi
- `COOLDOWN_PERIOD = 1.5` - Waktu tunggu dalam detik antar deteksi untuk mencegah eksekusi ganda

### Parameter Pelatihan di model.py

Konfigurasi proses training:

- `EPOCHS = 100` - Maksimal epoch training (dengan early stopping)
- `BATCH_SIZE = 16` - Jumlah sampel per batch
- Arsitektur model menggunakan 3 blok Conv2D dengan filter berturut-turut 32, 64, 128
- Callbacks: EarlyStopping dengan patience=15, ReduceLROnPlateau dengan patience=5

### Variabel Penting yang Harus Disesuaikan

1. Path ke file shortcut di folder apps/ harus sesuai dengan nama aplikasi yang terinstall di sistem
2. Threshold confidence dapat disesuaikan berdasarkan akurasi model hasil training
3. RMS threshold mungkin perlu disesuaikan berdasarkan sensitivitas mikrofon dan kondisi lingkungan noise
