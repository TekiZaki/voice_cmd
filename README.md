# VoiceCmD: Deep Learning Powered Voice Assistant

**Asisten suara berbasis Deep Learning untuk otomasi alur kerja Windows melalui kendali suara secara real-time.**

## Deskripsi Singkat

VoiceCmD adalah sistem asisten suara cerdas yang mampu mengenali perintah suara spesifik untuk menjalankan aplikasi atau mensimulasikan pintasan keyboard secara otomatis. Proyek ini mengatasi hambatan interaksi manual dengan menyediakan antarmuka "hands-free" menggunakan arsitektur Neural Network berbasis MFCC (Mel-frequency cepstrum coefficients) yang diimplementasikan dengan TensorFlow dan Keras.

## Fitur Inti

- **Pengenalan Suara Real-Time:** Memproses aliran audio secara langsung dengan latensi rendah untuk eksekusi perintah instan.
- **Otomasi Aplikasi & Shortcut:** Menjalankan file `.lnk` aplikasi Windows atau mensimulasikan kombinasi tombol keyboard (seperti `Ctrl+A`, `Alt+Tab`) melalui perintah suara.
- **HUD Interface Futuristik:** Antarmuka pengguna berbasis Tkinter dengan visualisasi gelombang audio (oscilloscope) dan log sistem yang informatif.
- **Data Acquisition Module:** Dilengkapi dengan alat pengumpul data audio terintegrasi untuk melatih perintah suara baru sesuai kebutuhan pengguna.
- **State Machine Awake/Standby:** Sistem cerdas yang hanya merespons perintah aktif setelah mendeteksi frase pemicu "Hello VoiceCmD", menghemat sumber daya sistem.

## Persyaratan & Instalasi Cepat

### Persyaratan

- Python 3.8+
- Perangkat input audio (Mikrofon)
- Windows OS (untuk dukungan fungsionalitas shortcut `.lnk` dan `pyautogui`)

### Instalasi

```powershell
# Kloning repositori
git clone https://github.com/TekiZaki/voice_cmd.git
cd voice_cmd

# Instalasi dependensi
pip install -r requirements.txt
```

## Contoh Penggunaan

Pastikan mikrofon Anda terhubung, lalu jalankan aplikasi utama untuk memulai asisten dalam mode standby.

```powershell
# Menjalankan asisten suara (HUD Terminal)
python main.py
```

_Gunakan frase "Hello VoiceCmD" untuk mengaktifkan asisten, kemudian ucapkan perintah yang telah terdaftar (contoh: "WhatsApp" atau "Note")._

## Konfigurasi Penting

- **`command_map.json`**: File utama untuk memetakan label suara ke aksi (path aplikasi atau kode tombol).
- **`MODELS_PATH`**: Lokasi penyimpanan file model `.h5` dan label encoder yang telah dilatih.

## Struktur Proyek

| File | Deskripsi Teknis |
| :--- | :--- |
| **audio_utils.py** | Modul pemrosesan sinyal digital yang bertanggung jawab untuk meningkatkan kualitas input audio. Implementasi mencakup reduksi noise berbasis algoritma stationary noise reduction, pre-emphasis untuk penguatan frekuensi tinggi guna memperjelas fitur wicara, normalisasi puncak untuk level volume yang konsisten, serta pemotongan otomatis bagian sunyi (trimming) menggunakan pustaka Librosa. |
| **data_collector.py** | Antarmuka grafis (GUI) berbasis Tkinter yang dirancang khusus untuk akuisisi dataset audio secara sistematis. Modul ini mendukung visualisasi sinyal waktu nyata dan memungkinkan pengguna untuk memetakan rekaman suara ke dua jenis aksi: eksekusi file shortcut Windows (.lnk) atau simulasi penekanan tombol keyboard (pyautogui). Setiap rekaman akan diproses secara otomatis melalui `audio_utils` sebelum disimpan ke direktori dataset. |
| **main.py** | Program utama yang menjalankan asisten suara dalam mode inferensi waktu nyata. Mengimplementasikan mesin status (state machine) Awake/Standby yang merespons frase pemicu "Hello VoiceCmD". Dilengkapi dengan antarmuka HUD (Heads-Up Display) futuristik yang menampilkan oscilloscope audio, log telemetri sistem, dan riwayat pengenalan perintah. Proses inferensi dilakukan secara efisien melalui threading untuk meminimalkan latensi eksekusi. |
| **model.py** | Skrip untuk manufaktur dan pelatihan model deep learning berbasis Neural Network Konvolusional (CNN). Modul ini melakukan ekstraksi fitur kompleks yang menggabungkan MFCC (Mel-frequency cepstrum coefficients) dengan Delta dan Delta-Delta guna menangkap karakteristik temporal suara. Strategi pelatihan mencakup augmentasi data (noise injection, time shifting, pitch shifting) dan penanganan background noise untuk memastikan model tetap tangguh dalam berbagai kondisi lingkungan. |

```text
voice_cmd/
├── apps/               # Pintasan (.lnk) aplikasi target
├── dataset/            # Kumpulan sampel audio untuk setiap label perintah
├── models/             # Artefak model terlatih (.h5) dan label encoder (.npy)
├── audio_utils.py      # Utilitas pengolahan sinyal audio
├── data_collector.py   # Modul akuisisi data dan konfigurasi perintah
├── main.py             # Entry point aplikasi utama dan HUD terminal
├── model.py            # Arsitektur model dan pipeline pelatihan AI
└── requirements.txt    # Daftar dependensi pustaka Python
```
