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

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(BASE_DIR, "apps")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MAP_FILE = os.path.join(BASE_DIR, "command_map.json")
SAMPLE_RATE = 44100
CHANNELS = 1

# Ensure directories exist
for d in [APPS_DIR, DATASET_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self._stream = None
        self.device_id = None
        self.waveform_data = np.zeros(100)
    
    def start_recording(self, device_id):
        self.device_id = device_id
        self.frames = []
        self.is_recording = True
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                self.frames.append(indata.copy())
                # Update waveform data for visualization (downsample/stride)
                self.waveform_data = indata[::len(indata)//100, 0] if len(indata) > 100 else indata[:, 0]

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=self.device_id,
            channels=CHANNELS,
            callback=callback
        )
        self._stream.start()

    def stop_recording(self, filename):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        if not self.frames:
            return False

        # Convert list of arrays to a single flat array
        audio_data = np.concatenate(self.frames, axis=0)
        # Convert to 16-bit PCM
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data_int16.tobytes())
        
        return True

class VoiceCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepVoice Data Collector")
        self.root.geometry("600x750")
        self.root.configure(bg="white")
        
        self.recorder = AudioRecorder()
        self.setup_styles()
        self.create_widgets()
        self.load_devices()
        self.load_mapping()
        
        self.update_waveform()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # General Styles
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white", foreground="black", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#00E676")

        # Entry/Combo Styles
        style.configure("TCombobox", fieldbackground="white", background="white", foreground="black")
        style.configure("TEntry", fieldbackground="white", foreground="black", insertcolor="black")

        # Button Styles
        style.configure("TButton", padding=10, font=("Segoe UI", 10, "bold"))
        style.configure("Action.TButton", background="#00E676", foreground="black")
        style.map("Action.TButton", background=[('active', '#00C853')])

        style.configure("Stop.TButton", background="#FF5252", foreground="white")
        style.map("Stop.TButton", background=[('active', '#D32F2F')])

    def create_widgets(self):
        main_container = ttk.Frame(self.root, padding="30")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_container, text="VOICE DATA COLLECTOR", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # Device Selection
        ttk.Label(main_container, text="Select Input Device:").pack(anchor=tk.W)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(main_container, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill=tk.X, pady=(5, 15))

        # Command Mapping Configuration
        config_frame = ttk.LabelFrame(main_container, text=" Command Configuration ", padding=15)
        config_frame.pack(fill=tk.X, pady=10)

        ttk.Label(config_frame, text="Command Label (e.g., buka_wa):").pack(anchor=tk.W)
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(config_frame, textvariable=self.label_var)
        self.label_entry.pack(fill=tk.X, pady=(5, 10))

        ttk.Label(config_frame, text="Target Shortcut or Keyboard Action:").pack(anchor=tk.W)
        shortcut_row = ttk.Frame(config_frame)
        shortcut_row.pack(fill=tk.X, pady=(5, 10))
        
        self.shortcut_var = tk.StringVar()
        self.shortcut_entry = ttk.Entry(shortcut_row, textvariable=self.shortcut_var)
        self.shortcut_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.btn_browse = ttk.Button(shortcut_row, text="Browse", command=self.browse_shortcut, width=8)
        self.btn_browse.pack(side=tk.RIGHT)

        # Action Type Toggle
        type_frame = ttk.Frame(config_frame)
        type_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.action_type = tk.StringVar(value="app")
        ttk.Radiobutton(type_frame, text="Application (.lnk)", variable=self.action_type, value="app", command=self.update_ui_mode).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(type_frame, text="Keyboard Action (key:)", variable=self.action_type, value="key", command=self.update_ui_mode).pack(side=tk.LEFT)

        # Waveform Canvas
        ttk.Label(main_container, text="Live Monitor:").pack(anchor=tk.W, pady=(10, 0))
        self.canvas = tk.Canvas(main_container, height=120, bg="white", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(fill=tk.X, pady=5)
        self.waveform_line = self.canvas.create_line(0, 60, 600, 60, fill="black", width=2)

        # Controls
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(main_container, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(pady=10)

        self.btn_record = ttk.Button(main_container, text="START RECORDING", style="Action.TButton", command=self.toggle_recording)
        self.btn_record.pack(fill=tk.X, pady=10)

        # Stats Table
        self.stats_text = tk.Text(main_container, height=6, bg="white", fg="black", font=("Consolas", 9), borderwidth=0)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.refresh_stats()

    def load_devices(self):
        devices = sd.query_devices()
        input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.device_combo['values'] = input_devices
        if input_devices:
            self.device_combo.current(0)

    def browse_shortcut(self):
        filename = filedialog.askopenfilename(
            initialdir=APPS_DIR,
            title="Select Windows Shortcut",
            filetypes=(("Shortcut files", "*.lnk"), ("All files", "*.*"))
        )
        if filename:
            self.shortcut_var.set(os.path.basename(filename))

    def update_ui_mode(self):
        """Toggle between app shortcut and keyboard shortcut mode."""
        if self.action_type.get() == "key":
            self.btn_browse.configure(state="disabled")
            if not self.shortcut_var.get().startswith("key:"):
                # Suggest a prefix or clear if it was an app shortcut
                self.shortcut_var.set("key:ctrl+a") 
        else:
            self.btn_browse.configure(state="normal")
            if self.shortcut_var.get().startswith("key:"):
                self.shortcut_var.set("")

    def toggle_recording(self):
        if not self.recorder.is_recording:
            label = self.label_var.get().strip()
            shortcut = self.shortcut_var.get().strip()
            
            if not label or not shortcut:
                messagebox.showwarning("Warning", "Please provide a Command Label and Shortcut/Action target.")
                return
            
            # Start
            try:
                device_idx = int(self.device_var.get().split(':')[0])
                self.recorder.start_recording(device_idx)
                self.btn_record.configure(text="STOP RECORDING (0s)", style="Stop.TButton")
                self.status_var.set(f"Status: Recording for [{label}]...")
                self.start_time = time.time()
                self.update_timer()
                self.save_mapping(label, shortcut)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start: {e}")
        else:
            # Stop
            label = self.label_var.get().strip()
            target_dir = os.path.join(DATASET_DIR, label)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(target_dir, f"{label}_{timestamp}.wav")
            
            if self.recorder.stop_recording(filename):
                self.status_var.set(f"Status: Saved record to {label} samples.")
            
            self.btn_record.configure(text="START RECORDING", style="Action.TButton")
            self.refresh_stats()

    def update_timer(self):
        if self.recorder.is_recording:
            elapsed = int(time.time() - self.start_time)
            self.btn_record.configure(text=f"STOP RECORDING ({elapsed}s)")
            self.root.after(1000, self.update_timer)

    def update_waveform(self):
        # Draw waveform on canvas
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        mid_y = height / 2
        
        data = self.recorder.waveform_data
        points = []
        
        step = max(1, width / len(data))
        for i, val in enumerate(data):
            x = i * step
            # Scale amplitude and amplify for visibility
            y = mid_y - (val * height * 0.8)
            points.extend([x, y])
        
        if len(points) >= 4:
            self.canvas.coords(self.waveform_line, *points)
        
        self.root.after(50, self.update_waveform)

    def load_mapping(self):
        if os.path.exists(MAP_FILE):
            try:
                with open(MAP_FILE, 'r') as f:
                    self.mapping = json.load(f)
            except:
                self.mapping = {}
        else:
            self.mapping = {}

    def save_mapping(self, label, shortcut):
        self.mapping[label] = shortcut
        with open(MAP_FILE, 'w') as f:
            json.dump(self.mapping, f, indent=4)

    def refresh_stats(self):
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, "DATASET STATISTICS:\n")
        self.stats_text.insert(tk.END, "-" * 30 + "\n")
        
        if not os.path.exists(DATASET_DIR):
            return
            
        folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
        for folder in folders:
            count = len(os.listdir(os.path.join(DATASET_DIR, folder)))
            self.stats_text.insert(tk.END, f" {folder.ljust(15)} : {count} samples\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceCollectorGUI(root)
    root.mainloop()
