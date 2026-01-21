import os
import json
import time
import threading
import queue
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pyautogui
import ctypes
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from audio_utils import enhance_audio

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
COMMAND_MAP_PATH = os.path.join(BASE_DIR, 'command_map.json')
APPS_PATH = os.path.join(BASE_DIR, 'apps')
SOUNDS_PATH = os.path.join(BASE_DIR, 'sound')

SAMPLE_RATE = 44100
DURATION = 1.0
N_MFCC = 40
RMS_THRESHOLD = 0.05
CONFIDENCE_THRESHOLD = 0.8
COOLDOWN_PERIOD = 1.5

# UI COLORS
BG_DARK = "#0F172A"
BG_PANEL = "#1E293B"
ACCENT_BLUE = "#3B82F6"
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"
ACCENT_GOLD = "#F59E0B"
TEXT_PRIMARY = "#F1F5F9"
TEXT_DIM = "#94A3B8"

class HUDPanel(tk.Canvas):
    """Custom Canvas with HUD-style borders."""
    def __init__(self, master, title="", **kwargs):
        kwargs.setdefault("bg", BG_DARK)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)
        self.title = title
        self.bind("<Configure>", self._draw_hud)

    def _draw_hud(self, event=None):
        self.delete("hud")
        w = self.winfo_width()
        h = self.winfo_height()
        
        # Border
        self.create_rectangle(2, 2, w-2, h-2, outline=BG_PANEL, width=1, tags="hud")
        
        # Corners (HUD style)
        cl = 15 # corner length
        # Top-Left
        self.create_line(2, 2, 2+cl, 2, fill=ACCENT_BLUE, width=2, tags="hud")
        self.create_line(2, 2, 2, 2+cl, fill=ACCENT_BLUE, width=2, tags="hud")
        # Top-Right
        self.create_line(w-2, 2, w-2-cl, 2, fill=ACCENT_BLUE, width=2, tags="hud")
        self.create_line(w-2, 2, w-2, 2+cl, fill=ACCENT_BLUE, width=2, tags="hud")
        # Bottom-Left
        self.create_line(2, h-2, 2+cl, h-2, fill=ACCENT_BLUE, width=2, tags="hud")
        self.create_line(2, h-2, 2, h-2-cl, fill=ACCENT_BLUE, width=2, tags="hud")
        # Bottom-Right
        self.create_line(w-2, h-2, w-2-cl, h-2, fill=ACCENT_BLUE, width=2, tags="hud")
        self.create_line(w-2, h-2, w-2, h-2-cl, fill=ACCENT_BLUE, width=2, tags="hud")

        # Title Background
        if self.title:
            self.create_rectangle(10, 0, 10 + (len(self.title)*8), 15, fill=BG_DARK, outline="", tags="hud")
            self.create_text(15, 7, text=self.title.upper(), fill=ACCENT_BLUE, font=("Consolas", 8, "bold"), anchor="w", tags="hud")

class VoiceAssistantCore:
    def __init__(self, log_queue, history_queue, state_callback):
        self.log_queue = log_queue
        self.history_queue = history_queue
        self.state_callback = state_callback
        
        self.is_running = False
        self.is_awake = False
        self.model = None
        self.le = None
        self.command_map = {}
        
        self.audio_buffer = np.zeros(int(DURATION * SAMPLE_RATE))
        self.waveform_data = np.zeros(100)
        self.last_action_time = 0

    def log(self, message, type="info"):
        self.log_queue.put((message, type))

    def load_resources(self):
        self.log("📦 Loading AI models and configurations...")
        try:
            model_path = os.path.join(MODELS_PATH, 'voice_model.h5')
            le_path = os.path.join(MODELS_PATH, 'label_encoder.npy')
            
            if not os.path.exists(model_path) or not os.path.exists(le_path):
                raise Exception("Model files not found. Please run model.py first.")
                
            self.model = tf.keras.models.load_model(model_path)
            self.le = LabelEncoder()
            self.le.classes_ = np.load(le_path, allow_pickle=True)
            
            if not os.path.exists(COMMAND_MAP_PATH):
                raise Exception(f"{COMMAND_MAP_PATH} not found.")
                
            with open(COMMAND_MAP_PATH, 'r') as f:
                self.command_map = json.load(f)
            
            self.log("✅ Resources loaded successfully.")
            return True
        except Exception as e:
            self.log(f"❌ Initialization Error: {e}", "error")
            return False

    def play_feedback(self, filename):
        def _play():
            path = os.path.join(SOUNDS_PATH, filename)
            if not os.path.exists(path): return
            alias = f"sound_{int(time.time() * 1000)}"
            try:
                ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, 0)
            except: pass
        threading.Thread(target=_play, daemon=True).start()

    def extract_features(self, audio):
        audio = enhance_audio(audio, SAMPLE_RATE)
        target_samples = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        return combined.T

    def execute_action(self, label):
        if label in self.command_map:
            action = self.command_map[label]
            if not action or action.lower() == "none": return
            
            if action.startswith("key:"):
                keys_str = action[4:].strip()
                keys = keys_str.split('+')
                self.log(f"⌨️ [AUTO] Keyboard Shortcut: {keys_str}", "success")
                try: pyautogui.hotkey(*keys)
                except Exception as e: self.log(f"❌ Keyboard Error: {e}", "error")
                return

            app_name = action
            app_path = os.path.join(APPS_PATH, app_name)
            if os.path.exists(app_path):
                self.log(f"🚀 [AUTO] Opening application: {app_name}", "success")
                try: os.startfile(app_path)
                except Exception as e: self.log(f"❌ Open Error: {e}", "error")
            else:
                self.log(f"⚠️ Shortcut file not found: {app_name}", "warning")

    def audio_callback(self, indata, frames, time_info, status):
        # Shift buffer and add new data
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata.flatten()
        # Update small visualizer data
        self.waveform_data = indata[::len(indata)//100, 0] if len(indata) > 100 else indata[:, 0]

    def run_inference_loop(self):
        self.is_running = True
        self.play_feedback('standby.mp3')
        
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self.audio_callback):
                while self.is_running:
                    current_time = time.time()
                    if current_time - self.last_action_time < COOLDOWN_PERIOD:
                        time.sleep(0.1)
                        continue

                    check_len = int(0.15 * SAMPLE_RATE)
                    rms = np.sqrt(np.mean(self.audio_buffer[-check_len:]**2))
                    
                    if rms > RMS_THRESHOLD:
                        if self.is_awake:
                            self.log(f"✨ Sound Detected (RMS: {rms:.3f}) - Analyzing...", "debug")
                        
                        time.sleep(0.6) 
                        snapshot = self.audio_buffer.copy()
                        
                        try:
                            features = self.extract_features(snapshot)
                            input_data = features[np.newaxis, ...]
                            predictions = self.model.predict(input_data, verbose=0)
                            top_idx = np.argmax(predictions[0])
                            confidence = predictions[0][top_idx]
                            label = self.le.classes_[top_idx]
                            
                            if label == "background":
                                pass
                            elif confidence < CONFIDENCE_THRESHOLD:
                                if self.is_awake:
                                    self.log(f"❓ Low confidence detection: {label} ({confidence*100:.1f}%)", "warning")
                            else:
                                if not self.is_awake:
                                    if label == "hello_voicecmd":
                                        self.is_awake = True
                                        self.state_callback(True)
                                        self.log("💡 System AWAKE", "success")
                                        self.play_feedback('active.mp3')
                                        self.last_action_time = time.time()
                                else:
                                    if label == "sleep_cmd":
                                        self.is_awake = False
                                        self.state_callback(False)
                                        self.log("😴 System STANDBY", "info")
                                        self.play_feedback('standby.mp3')
                                        self.last_action_time = time.time()
                                    else:
                                        self.log(f"🎯 MATCH: {label.upper()} ({confidence*100:.1f}%)", "success")
                                        self.history_queue.put((label, confidence))
                                        self.execute_action(label)
                                        self.last_action_time = time.time()
                        
                        except Exception as e:
                            self.log(f"❌ Prediction Error: {e}", "error")
                    
                    time.sleep(0.05)
        except Exception as e:
            self.log(f"💥 Fatal Audio Error: {e}", "error")
            self.is_running = False

class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DEEPVOICE AI - HUD TERMINAL")
        self.root.geometry("1000x800")
        self.root.configure(bg=BG_DARK)
        
        self.log_queue = queue.Queue()
        self.history_queue = queue.Queue()
        
        self.core = VoiceAssistantCore(self.log_queue, self.history_queue, self.update_state_ui)
        self.setup_ui()
        
        if self.core.load_resources():
            self.inference_thread = threading.Thread(target=self.core.run_inference_loop, daemon=True)
            self.inference_thread.start()
        
        self.process_queues()
        self.animate_wf()

    def setup_ui(self):
        # Header
        self.header = HUDPanel(self.root, title="System Header", height=60)
        self.header.pack(fill=tk.X, padx=10, pady=(10, 5))
        self.header.create_text(20, 30, text="DEEPVOICE HUD v2.0", fill=ACCENT_BLUE, font=("Consolas", 16, "bold"), anchor="w")
        self.uptime_id = self.header.create_text(980, 30, text="UPTIME: 00:00:00", fill=TEXT_DIM, font=("Consolas", 10), anchor="e")
        self.start_time = time.time()

        # Main Layout (Left: Stats/Vis, Right: History)
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        left_frame = tk.Frame(main_frame, bg=BG_DARK)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # State Indicator
        self.state_panel = HUDPanel(left_frame, title="System State", height=120)
        self.state_panel.pack(fill=tk.X, pady=(0, 5))
        self.state_text = self.state_panel.create_text(250, 60, text="STANDBY", fill=TEXT_DIM, font=("Consolas", 32, "bold"))
        self.state_circle = self.state_panel.create_oval(30, 30, 90, 90, outline=TEXT_DIM, width=4)
        
        # Visualizer
        self.vis_panel = HUDPanel(left_frame, title="Oscilloscope / Audio Feed", height=200)
        self.vis_panel.pack(fill=tk.X, pady=5)
        self.vis_line = self.vis_panel.create_line(0, 100, 500, 100, fill=ACCENT_BLUE, width=2)

        # System Logs
        log_panel = HUDPanel(left_frame, title="System Telemetry Logs", height=250)
        log_panel.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.log_text = tk.Text(log_panel, bg=BG_DARK, fg=TEXT_PRIMARY, font=("Consolas", 9), relief=tk.FLAT, borderwidth=0, padx=15, pady=15)
        self.log_text.place(x=5, y=20, relwidth=0.98, relheight=0.9)
        
        # Right Pane: History
        right_frame = tk.Frame(main_frame, bg=BG_DARK, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        hist_panel = HUDPanel(right_frame, title="Recognition History", width=340)
        hist_panel.pack(fill=tk.BOTH, expand=True)
        self.hist_text = tk.Text(hist_panel, bg=BG_DARK, fg=TEXT_PRIMARY, font=("Consolas", 10), relief=tk.FLAT, borderwidth=0, padx=15, pady=20)
        self.hist_text.place(x=5, y=30, relwidth=0.95, relheight=0.9)

        # Tag colors
        self.log_text.tag_configure("success", foreground=ACCENT_GREEN)
        self.log_text.tag_configure("error", foreground=ACCENT_RED)
        self.log_text.tag_configure("warning", foreground=ACCENT_GOLD)
        self.log_text.tag_configure("debug", foreground=TEXT_DIM)
        
        self.hist_text.tag_configure("label", foreground=ACCENT_BLUE, font=("Consolas", 10, "bold"))
        self.hist_text.tag_configure("conf", foreground=ACCENT_GREEN)

    def update_state_ui(self, is_awake):
        if is_awake:
            self.state_panel.itemconfig(self.state_text, text="ACTIVE", fill=ACCENT_BLUE)
            self.state_panel.itemconfig(self.state_circle, outline=ACCENT_BLUE)
        else:
            self.state_panel.itemconfig(self.state_text, text="STANDBY", fill=TEXT_DIM)
            self.state_panel.itemconfig(self.state_circle, outline=TEXT_DIM)

    def animate_wf(self):
        # Simple uptime counter
        elapsed = int(time.time() - self.start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        # For simplicity, we can't easily find the item ID for the header text across classes without a ref
        # but we'll try to find it or just skip for now.
        
        # Audio Waveform
        w = self.vis_panel.winfo_width()
        h_vis = self.vis_panel.winfo_height()
        mid_y = h_vis / 2
        
        # Uptime Update
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self.header.itemconfig(self.uptime_id, text=f"UPTIME: {h:02d}:{m:02d}:{s:02d}")

        data = self.core.waveform_data
        points = []
        step = max(1, w / len(data))
        
        # Animation pulse for circle if awake
        if self.core.is_awake:
            pulse = (int(time.time() * 10) % 5) + 2
            self.state_panel.itemconfig(self.state_circle, width=pulse, outline=ACCENT_BLUE)
            # Add a slight shadow/glow effect by updating another item if we had one
        else:
            self.state_panel.itemconfig(self.state_circle, width=2, outline=TEXT_DIM)

        for i, val in enumerate(data):
            x = i * step
            # Scale amplitude and amplify
            y = mid_y - (val * h_vis * 1.5) # Increased scale for more "activity"
            points.extend([x, y])
        
        if len(points) >= 4:
            self.vis_panel.coords(self.vis_line, *points)
            
        # Draw a scanning line
        scan_x = (time.time() * 150) % w
        if not hasattr(self, 'scan_line'):
            self.scan_line = self.vis_panel.create_line(scan_x, 0, scan_x, h_vis, fill=ACCENT_BLUE, dash=(4, 4), tags="hud")
        else:
            self.vis_panel.coords(self.scan_line, scan_x, 0, scan_x, h_vis)
        
        self.root.after(40, self.animate_wf)

    def process_queues(self):
        # Process Logs
        try:
            while True:
                msg, tag = self.log_queue.get_nowait()
                ts = datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{ts}] ", "debug")
                self.log_text.insert(tk.END, f"{msg}\n", tag)
                self.log_text.see(tk.END)
                self.log_queue.task_done()
        except queue.Empty:
            pass

        # Process History
        try:
            while True:
                label, conf = self.history_queue.get_nowait()
                ts = datetime.now().strftime("%H:%M:%S")
                self.hist_text.insert(tk.END, f"• {ts} ", "debug")
                self.hist_text.insert(tk.END, f"{label.upper()}\n", "label")
                self.hist_text.insert(tk.END, f"  Confidence: {conf*100:.1f}%\n\n", "conf")
                self.hist_text.see(tk.END)
                self.history_queue.task_done()
        except queue.Empty:
            pass

        self.root.after(100, self.process_queues)

if __name__ == "__main__":
    root = tk.Tk()
    # Simple trick for removing title bar if you wanted true HUD, but standard is safer
    # root.overrideredirect(True) 
    app = VoiceAssistantGUI(root)
    
    def on_closing():
        app.core.is_running = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
