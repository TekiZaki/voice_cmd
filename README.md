# VoiceCmD: AI-Powered Voice Assistant

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

```text
voice_cmd/
├── apps/               # Pintasan (.lnk) aplikasi yang akan dijalankan
├── dataset/            # Data audio untuk pelatihan model
├── models/             # Model AI (.h5) dan Label Encoder
├── audio_utils.py      # Pengolahan sinyal dan peningkatan kualitas audio
├── data_collector.py   # GUI untuk merekam dataset baru
├── main.py             # Aplikasi utama dengan HUD Interface
├── model.py            # Arsitektur Neural Network dan skrip pelatihan
└── requirements.txt    # Daftar pustaka Python yang diperlukan
```
