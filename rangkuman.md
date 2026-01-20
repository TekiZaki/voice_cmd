Berikut adalah deskripsi teknis lengkap dalam bentuk poin-poin yang bisa Anda gunakan untuk dokumentasi proyek, laporan, dan panduan pengerjaan kelompok Anda.

---

### 1. Struktur Folder (Project Tree)

Struktur ini dirancang agar modular. Data suara, model AI, dan shortcut aplikasi terpisah dengan rapi.

```text
VoiceCmd_Project/
│
├── apps/                        # [FOLDER PENTING] Tempat menyimpan shortcut
│   ├── wa.lnk                   # Shortcut ke WhatsApp
│   ├── edge.lnk                 # Shortcut ke Microsoft Edge
│   └── myapp.lnk                # Shortcut ke Aplikasi NeutralinoJS Anda
│
├── dataset/                     # [FOLDER OTOMATIS] Hasil rekaman data_collector.py
│   ├── buka_wa/                 # Berisi ratusan file .wav
│   ├── buka_edge/               # Berisi ratusan file .wav
│   └── ...
│
├── models/                      # Tempat menyimpan hasil training
│   ├── voice_model.h5           # File otak AI (Weights)
│   └── label_encoder.npy        # File kamus label (Angka ke Kata)
│
├── command_map.json             # Database mapping (Label Suara -> Nama Shortcut)
├── data_collector.py            # Script untuk merekam suara
├── model.py                     # Script untuk melatih AI (Training)
├── main.py                      # Script utama Voice Assistant (Eksekusi)
└── requirements.txt             # Daftar library (tensorflow, librosa, dll)

```

---

### 2. Deskripsi `data_collector.py` (Perekam & Mapping)

Script ini berfungsi sebagai **GUI Tool** untuk mengumpulkan data latih sekaligus mengatur logika perintah.

- **Fungsi Utama:**
- Merekam suara pengguna melalui mikrofon laptop/eksternal.
- Membuat folder dataset secara otomatis berdasarkan label yang diinput.
- Memetakan variasi ucapan ke satu file shortcut yang sama.

- **Fitur Teknis:**
- **Audio Settings:** Sample Rate 44.1kHz, Mono Channel, Format `.wav` lossless.
- **Mapping System (JSON):** Setiap kali merekam label baru, script otomatis mengupdate file `command_map.json`.
- _Contoh:_ Input Label "buka_whatsapp" -> Input Target "wa.lnk".

- **Threading:** Menggunakan _thread_ terpisah untuk proses perekaman agar GUI tidak _freeze_ (macet) saat sedang merekam.
- **Auto-Naming:** File disimpan dengan format `label_TIMESTAMP.wav` untuk mencegah file tertimpa.

---

### 3. Deskripsi `model.py` (Pelatihan Deep Learning)

Script ini adalah "dapur" tempat dataset suara diolah menjadi model cerdas. Script ini hanya dijalankan saat Anda selesai mengumpulkan data atau ingin update akurasi.

- **1. Preprocessing (Pengolahan Data Awal)**
- **Librosa Loading:** Membaca semua file `.wav` dari folder `dataset/`.
- **Feature Extraction (MFCC):** Mengubah gelombang suara menjadi **MFCC (Mel-Frequency Cepstral Coefficients)**.
- _Penjelasan Laporan:_ MFCC mengubah audio menjadi representasi numerik yang mirip dengan cara telinga manusia mendengar frekuensi. Ini adalah "gambar" yang akan dibaca oleh model.

- **Label Encoding:** Mengubah nama folder (String) menjadi Angka (Integer).
- Contoh: `noise`=0, `buka_whatsapp`=1, `buka_edge`=2.

- **2. Arsitektur Model (CNN)**
- **Input Layer:** Menerima matriks MFCC (misal ukuran 40x174).
- **Conv2D Layer:** Melakukan _scanning_ untuk mencari pola unik pada frekuensi suara.
- **MaxPooling2D:** Memperkecil ukuran data untuk mengambil fitur yang paling menonjol (mengurangi beban komputasi).
- **Dropout:** Mematikan sebagian neuron secara acak saat latihan untuk mencegah _Overfitting_ (model menghafal data, bukan belajar pola).
- **Flatten:** Meratakan data matriks menjadi satu baris vektor panjang.
- **Dense (Softmax):** Lapisan terakhir yang mengeluarkan probabilitas (persentase) untuk setiap kelas perintah.

- **3. Output Script**
- Setelah proses _training_ (misal 50 epoch) selesai, script akan menghasilkan dua file di folder `models/`:

1. `voice_model.h5`: File model Deep Learning yang sudah jadi.
2. `label_encoder.npy`: File kunci jawaban untuk menerjemahkan kembali prediksi angka (0, 1, 2) menjadi teks label asli.

---

### Rangkuman Alur Kerja untuk Laporan

Jika dosen bertanya "Bagaimana data mengalir di sistem Anda?", Anda bisa menjawab berdasarkan poin-poin di atas:

1. **Input:** Suara direkam via `data_collector.py` dan disimpan sebagai `.wav`.
2. **Mapping:** `data_collector.py` mencatat bahwa suara "X" bertujuan membuka shortcut "Y" di `command_map.json`.
3. **Training:** `model.py` mengambil file `.wav`, mengubahnya jadi MFCC, melatih otak CNN, dan menyimpan hasilnya.
4. **Eksekusi (Nanti di main.py):** Program utama memuat model, mendengar suara mic, memprediksi label, mengecek `command_map.json`, lalu membuka shortcut di folder `apps/`.
