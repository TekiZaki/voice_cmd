import os
import wave
import json
import time
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from audio_utils import enhance_audio

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Direktori aplikasi
APPS_DIR = os.path.join(BASE_DIR, "apps")              # Direktori aplikasi shortcut
DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # Direktori dataset audio
MAP_FILE = os.path.join(BASE_DIR, "command_map.json")  # File pemetaan perintah
SAMPLE_RATE = 44100  # Tingkat sampling audio
CHANNELS = 1         # Jumlah channel audio (mono)

# --- WARNA & ESTETIKA UI ---
BG_DARK = "#0F172A"       # Latar belakang gelap utama
BG_CARD = "#1E293B"       # Latar panel/card
BG_HIGHLIGHT = "#334155"  # Warna highlight
ACCENT_BLUE = "#38BDF8"   # Aksen biru neon
ACCENT_CYAN = "#22D3EE"   # Aksen cyan neon
ACCENT_GREEN = "#10B981"  # Hijau (sukses)
RECORD_RED = "#FB7185"    # Merah/pink neon (recording)
TEXT_MAIN = "#F8FAFC"     # Teks utama putih
TEXT_MUTED = "#94A3B8"    # Teks redup
BORDER_COLOR = "#334155"  # Warna border

# Membuat direktori jika belum ada
for d in [APPS_DIR, DATASET_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

class AudioRecorder:
    """Kelas untuk merekam audio dari microphone."""
    def __init__(self):
        self.is_recording = False  # Status rekaman
        self.frames = []           # Buffer frame audio
        self._stream = None        # Stream audio
        self.device_id = None      # ID perangkat input
        self.waveform_data = np.zeros(200)  # Data untuk visualisasi waveform
    
    def start_recording(self, device_id):
        """Memulai perekaman audio."""
        self.device_id = device_id
        self.frames = []
        self.is_recording = True
        
        def callback(indata, frames, time, status):
            """Callback untuk menerima data audio dari stream."""
            if self.is_recording:
                self.frames.append(indata.copy())
                # Update data waveform untuk visualisasi
                current_data = indata[:, 0]
                stride = max(1, len(current_data) // 100)
                self.waveform_data = current_data[::stride]

        # Membuka stream input audio
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=self.device_id,
            channels=CHANNELS,
            callback=callback
        )
        self._stream.start()

    def stop_recording(self, filename):
        """Menghentikan perekaman dan menyimpan ke file WAV."""
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        if not self.frames:
            return False

        # Menggabungkan semua frame audio
        audio_data = np.concatenate(self.frames, axis=0).flatten()
        
        # Perbaikan kualitas audio
        try:
            audio_data = enhance_audio(audio_data, SAMPLE_RATE)
        except:
            pass  # Fallback jika enhancement gagal

        # Konversi ke format int16 untuk WAV
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Menyimpan ke file WAV
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data_int16.tobytes())
        
        return True

class RoundedFrame(tk.Canvas):
    """Frame kustom dengan sudut membulat menggunakan Canvas."""
    def __init__(self, parent, radius=20, bg=BG_DARK, border_color=BORDER_COLOR, **kwargs):
        super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)
        self.radius = radius
        self.border_color = border_color
        self.bind("<Configure>", self._draw)

    def _draw(self, event=None):
        """Menggambar frame dengan sudut membulat."""
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        r = self.radius
        
        # Menggambar bentuk dengan sudut membulat
        self.create_rounded_rect(2, 2, w-2, h-2, r, fill=BG_CARD, outline=self.border_color, width=1)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        """Membuat polygon dengan sudut membulat."""
        points = [
            x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1,
        ]
        return self.create_polygon(*points, smooth=True, **kwargs)

class VoiceCollectorGUI:
    """Antarmuka pengguna untuk pengumpulan data suara."""
    def __init__(self, root):
        self.root = root
        self.root.title("DATA COLLECTOR v2.0 - CYBER HUD")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG_DARK)
        
        self.recorder = AudioRecorder()  # Instance perekam audio
        self.pulse_val = 0               # Nilai untuk animasi pulse
        self.pulse_dir = 1               # Arah animasi pulse
        
        self.setup_ui()
        self.load_devices()
        self.load_mapping()
        self.update_loop()

    def setup_ui(self):
        """Membangun antarmuka pengguna."""
        # Header Atas
        header_frame = tk.Frame(self.root, bg=BG_DARK, height=80)
        header_frame.pack(fill=tk.X, padx=30, pady=(20, 10))
        
        title_label = tk.Label(header_frame, text="SYSTEM DATA ACQUISITION", font=("Segoe UI", 24, "bold"), fg=TEXT_MAIN, bg=BG_DARK)
        title_label.pack(side=tk.LEFT)
        
        version_label = tk.Label(header_frame, text="MODULE: VOICE_CORE_V2", font=("Consolas", 10), fg=ACCENT_BLUE, bg=BG_DARK)
        version_label.pack(side=tk.LEFT, padx=20, pady=(12, 0))

        # Frame Konten Utama
        self.content_frame = tk.Frame(self.root, bg=BG_DARK)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Kolom Kiri: Kontrol & Waveform (65%)
        self.left_col = tk.Frame(self.content_frame, bg=BG_DARK)
        self.left_col.place(relx=0, rely=0, relwidth=0.65, relheight=1)

        # Kolom Kanan: Log Sistem (35%)
        self.right_col = tk.Frame(self.content_frame, bg=BG_DARK)
        self.right_col.place(relx=0.67, rely=0, relwidth=0.33, relheight=1)

        self.setup_left_col()
        self.setup_right_col()

    def setup_left_col(self):
        """Membangun kolom kiri (kontrol dan visualisasi)."""
        # Bagian Pemilihan Perangkat
        dev_card = RoundedFrame(self.left_col, radius=15)
        dev_card.place(relx=0, rely=0, relwidth=1, height=100)
        
        tk.Label(dev_card, text="INPUT SOURCE", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).place(x=20, y=15)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(dev_card, textvariable=self.device_var, state="readonly")
        self.device_combo.place(x=20, y=45, relwidth=0.9, height=35)
        
        # Bagian Konfigurasi Perintah
        cfg_card = RoundedFrame(self.left_col, radius=15)
        cfg_card.place(relx=0, rely=0.15, relwidth=1, height=220)
        
        tk.Label(cfg_card, text="COMMAND CONFIGURATION", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).place(x=20, y=15)
        
        # Input Label Perintah
        tk.Label(cfg_card, text="COMMAND LABEL", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).place(x=20, y=45)
        self.label_var = tk.StringVar()
        self.label_entry = tk.Entry(cfg_card, textvariable=self.label_var, bg=BG_HIGHLIGHT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN, borderwidth=0)
        self.label_entry.place(x=20, y=65, relwidth=0.9, height=30)

        # Input Aksi/Shortcut
        tk.Label(cfg_card, text="TARGET ACTION / SHORTCUT", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).place(x=20, y=105)
        self.shortcut_var = tk.StringVar()
        self.shortcut_entry = tk.Entry(cfg_card, textvariable=self.shortcut_var, bg=BG_HIGHLIGHT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN, borderwidth=0)
        self.shortcut_entry.place(x=20, y=125, relwidth=0.7, height=30)
        
        self.btn_browse = tk.Button(cfg_card, text="BROWSE", font=("Segoe UI", 8, "bold"), bg=ACCENT_BLUE, fg=BG_DARK, activebackground=ACCENT_CYAN, command=self.browse_shortcut, relief=tk.FLAT)
        self.btn_browse.place(relx=0.75, y=125, relwidth=0.2, height=30)

        # Pilihan Mode Aksi
        self.action_type = tk.StringVar(value="app")
        tk.Radiobutton(cfg_card, text="APP (.LNK)", variable=self.action_type, value="app", bg=BG_CARD, fg=TEXT_MAIN, activebackground=BG_CARD, selectcolor=BG_DARK, command=self.update_ui_mode).place(x=20, y=170)
        tk.Radiobutton(cfg_card, text="KEYBOARD (KEY:)", variable=self.action_type, value="key", bg=BG_CARD, fg=TEXT_MAIN, activebackground=BG_CARD, selectcolor=BG_DARK, command=self.update_ui_mode).place(x=130, y=170)

        # Bagian Visualisasi Waveform
        self.wf_container = RoundedFrame(self.left_col, radius=15)
        self.wf_container.place(relx=0, rely=0.47, relwidth=1, height=180)
        
        tk.Label(self.wf_container, text="LIVE SIGNAL MONITOR", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).place(x=20, y=15)
        
        self.canvas = tk.Canvas(self.wf_container, bg=BG_CARD, highlightthickness=0)
        self.canvas.place(x=20, y=45, relwidth=0.92, height=110)
        self.wf_line_glow = self.canvas.create_line(0, 55, 600, 55, fill=ACCENT_BLUE, width=4)
        self.wf_line = self.canvas.create_line(0, 55, 600, 55, fill=TEXT_MAIN, width=1.5)

        # Bagian Tombol Rekam
        self.rec_section = RoundedFrame(self.left_col, radius=15)
        self.rec_section.place(relx=0, rely=0.74, relwidth=1, height=120)

        self.btn_record = tk.Button(self.rec_section, text="START ACQUISITION", font=("Segoe UI", 12, "bold"), bg=ACCENT_GREEN, fg=BG_DARK, activebackground=ACCENT_CYAN, relief=tk.FLAT, command=self.toggle_recording)
        self.btn_record.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.6)
        
        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.status_label = tk.Label(self.left_col, textvariable=self.status_var, font=("Consolas", 9), fg=TEXT_MUTED, bg=BG_DARK)
        self.status_label.place(relx=0, rely=0.92)

    def setup_right_col(self):
        """Membangun kolom kanan (log data sistem)."""
        log_card = RoundedFrame(self.right_col, radius=15)
        log_card.place(relx=0, rely=0, relwidth=1, relheight=1)

        tk.Label(log_card, text="SYSTEM DATA LOG", font=("Segoe UI", 11, "bold"), fg=ACCENT_BLUE, bg=BG_CARD).place(x=20, y=20)

        # Container untuk scrolling
        container = tk.Frame(log_card, bg=BG_CARD)
        container.place(x=20, y=60, relwidth=0.88, relheight=0.85)

        # Canvas untuk scrolling
        self.stats_canvas = tk.Canvas(container, bg=BG_CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.stats_canvas.yview, 
                                bg=BG_HIGHLIGHT, troughcolor=BG_CARD, 
                                activebackground=ACCENT_BLUE, width=8)
        
        # Frame scrollable di dalam canvas
        self.stats_frame = tk.Frame(self.stats_canvas, bg=BG_CARD)
        
        # Membuat window di canvas
        self.stats_window = self.stats_canvas.create_window((0, 0), window=self.stats_frame, anchor="nw")
        
        # Konfigurasi scroll
        self.stats_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack elemen
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind configure untuk update scroll region
        self.stats_frame.bind("<Configure>", lambda e: self.stats_canvas.configure(scrollregion=self.stats_canvas.bbox("all")))
        
        # Bind mousewheel untuk smooth scrolling
        def _on_mousewheel(event):
            self.stats_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.stats_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.refresh_stats()

    def update_loop(self):
        """Loop update untuk animasi dan visualisasi."""
        # Update waveform
        self.draw_waveform()
        
        # Animasi pulse saat recording
        if self.recorder.is_recording:
            self.pulse_val += 2 * self.pulse_dir
            if self.pulse_val >= 30 or self.pulse_val <= 0:
                self.pulse_dir *= -1
            
            # Efek glow pada bagian recording
            glow_alpha = int(self.pulse_val)
            self.rec_section.config(highlightthickness=2, highlightbackground=RECORD_RED, highlightcolor=RECORD_RED)
            self.status_label.config(fg=RECORD_RED)
        else:
            self.rec_section.config(highlightthickness=0)
            self.status_label.config(fg=TEXT_MUTED)
        
        self.root.after(30, self.update_loop)

    def draw_waveform(self):
        """Menggambar visualisasi waveform audio."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        mid_y = h / 2
        data = self.recorder.waveform_data
        
        if len(data) < 2: return
        
        points = []
        step = w / (len(data) - 1)
        for i, val in enumerate(data):
            x = i * step
            y = mid_y - (val * h * 0.9)  # Amplifikasi untuk visual
            points.extend([x, y])
            
        # Warna berubah saat recording
        color = RECORD_RED if self.recorder.is_recording else ACCENT_BLUE
        glow_color = "#4c1d24" if self.recorder.is_recording else "#1e3a4c"
        
        self.canvas.coords(self.wf_line_glow, *points)
        self.canvas.itemconfig(self.wf_line_glow, fill=glow_color, width=6)
        self.canvas.coords(self.wf_line, *points)
        self.canvas.itemconfig(self.wf_line, fill=color)

    def load_devices(self):
        """Memuat daftar perangkat audio input."""
        devices = sd.query_devices()
        input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.device_combo['values'] = input_devices
        if input_devices:
            self.device_combo.current(0)

    def browse_shortcut(self):
        """Membuka dialog untuk memilih file shortcut."""
        filename = filedialog.askopenfilename(initialdir=APPS_DIR, title="Select Windows Shortcut", filetypes=(("Shortcut files", "*.lnk"), ("All files", "*.*")))
        if filename: 
            self.shortcut_var.set(os.path.basename(filename))

    def update_ui_mode(self):
        """Memperbarui UI berdasarkan mode aksi yang dipilih."""
        if self.action_type.get() == "key":
            self.btn_browse.configure(state="disabled")
            if not self.shortcut_var.get().startswith("key:"): 
                self.shortcut_var.set("key:ctrl+a") 
        else:
            self.btn_browse.configure(state="normal")
            if self.shortcut_var.get().startswith("key:"): 
                self.shortcut_var.set("")

    def toggle_recording(self):
        """Toggle perekaman (mulai/berhenti)."""
        if not self.recorder.is_recording:
            # Memulai perekaman
            label = self.label_var.get().strip()
            shortcut = self.shortcut_var.get().strip()
            if not label or not shortcut:
                messagebox.showwarning("Warning", "Configuration incomplete.")
                return
            
            try:
                device_idx = int(self.device_var.get().split(':')[0])
                self.recorder.start_recording(device_idx)
                self.btn_record.configure(text="STOPPING ACQUISITION...", bg=RECORD_RED)
                self.status_var.set(f"RECORDING DATA: {label.upper()}")
                self.start_time = time.time()
                self.update_timer()
                self.save_mapping(label, shortcut)
            except Exception as e:
                messagebox.showerror("Error", f"Link error: {e}")
        else:
            # Menghentikan perekaman
            label = self.label_var.get().strip()
            target_dir = os.path.join(DATASET_DIR, label)
            if not os.path.exists(target_dir): 
                os.makedirs(target_dir)
            
            filename = os.path.join(target_dir, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            if self.recorder.stop_recording(filename):
                self.status_var.set(f"DATA STORED: {label.upper()} +1")
            
            self.btn_record.configure(text="START ACQUISITION", bg=ACCENT_GREEN)
            self.refresh_stats()

    def update_timer(self):
        """Memperbarui timer saat recording."""
        if self.recorder.is_recording:
            elapsed = int(time.time() - self.start_time)
            self.btn_record.configure(text=f"TERMINATE ({elapsed}s)")
            self.root.after(1000, self.update_timer)

    def load_mapping(self):
        """Memuat pemetaan perintah dari file JSON."""
        if os.path.exists(MAP_FILE):
            try:
                with open(MAP_FILE, 'r') as f: 
                    self.mapping = json.load(f)
            except: 
                self.mapping = {}
        else: 
            self.mapping = {}

    def save_mapping(self, label, shortcut):
        """Menyimpan pemetaan perintah ke file JSON."""
        self.mapping[label] = shortcut
        with open(MAP_FILE, 'w') as f: 
            json.dump(self.mapping, f, indent=4)

    def select_dataset(self, label):
        """Mengisi field konfigurasi otomatis saat dataset diklik."""
        shortcut = self.mapping.get(label, "")
        
        # Set tipe aksi terlebih dahulu
        if shortcut.startswith("key:"):
            self.action_type.set("key")
        else:
            self.action_type.set("app")
            
        self.label_var.set(label)
        self.shortcut_var.set(shortcut)
        self.update_ui_mode()

    def refresh_stats(self):
        """Memperbarui tampilan statistik dataset."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        if not os.path.exists(DATASET_DIR): return
            
        folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
        
        for i, folder in enumerate(folders):
            count = len(os.listdir(os.path.join(DATASET_DIR, folder)))
            
            row = tk.Frame(self.stats_frame, bg=BG_HIGHLIGHT if i % 2 == 0 else BG_CARD, pady=8, cursor="hand2")
            row.pack(fill=tk.X)
            
            # Format nama untuk tampilan
            display_name = folder.replace('_', ' ').title()
            
            lbl_name = tk.Label(row, text=display_name, font=("Consolas", 10, "bold"), fg=TEXT_MAIN, bg=row['bg'])
            lbl_name.pack(side=tk.LEFT, padx=15)
            
            lbl_count = tk.Label(row, text=f"{count} SAMPLES", font=("Consolas", 9), fg=ACCENT_BLUE, bg=row['bg'])
            lbl_count.pack(side=tk.RIGHT, padx=15)

            # Bind klik untuk memilih dataset
            for widget in (row, lbl_name, lbl_count):
                widget.bind("<Button-1>", lambda e, f=folder: self.select_dataset(f))

if __name__ == "__main__":
    root = tk.Tk()
    
    # Konfigurasi style ttk untuk combobox
    style = ttk.Style()
    style.theme_use('clam')
    
    # Konfigurasi field combobox
    style.configure("TCombobox", 
                    fieldbackground=BG_HIGHLIGHT,
                    background=BG_CARD,
                    foreground=TEXT_MAIN,
                    arrowcolor=ACCENT_BLUE,
                    borderwidth=1,
                    relief="flat")
    
    # Konfigurasi dropdown listbox
    style.map('TCombobox',
              fieldbackground=[('readonly', BG_HIGHLIGHT)],
              selectbackground=[('readonly', ACCENT_BLUE)],
              selectforeground=[('readonly', BG_DARK)],
              foreground=[('readonly', TEXT_MAIN)])
    
    # Style untuk dropdown list
    root.option_add('*TCombobox*Listbox.background', BG_CARD)
    root.option_add('*TCombobox*Listbox.foreground', TEXT_MAIN)
    root.option_add('*TCombobox*Listbox.selectBackground', ACCENT_BLUE)
    root.option_add('*TCombobox*Listbox.selectForeground', BG_DARK)
    root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))
    
    app = VoiceCollectorGUI(root)
    root.mainloop()
