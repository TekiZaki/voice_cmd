# Task: Implement Data Collector GUI

Build a Python-based GUI application for recording voice samples and mapping them to application shortcuts.

## 1. Analysis

- **Goal**: Create `data_collector.py` to build a dataset for voice command training.
- **Requirements**:
  - **UI**: Tkinter with a modern dark theme (visceral appeal).
  - **Features**:
    - Device selection (dropdown).
    - Manual record (Start/Stop).
    - Label management (input field).
    - Shortcut mapping (file browser for `.lnk` in `apps/`).
    - Live Waveform visualizer (Tkinter Canvas).
    - Data persistence: `command_map.json` and `.wav` files in `dataset/{label}/`.
- **Tech Stack**: Python, `tkinter`, `sounddevice`, `numpy`, `wave`, `json`, `threading`.

## 2. Plan

1. **Infrastructure**:
   - Ensure `apps/`, `dataset/` directories exist.
   - Setup `command_map.json` initialization.
2. **Audio Engine**:
   - Use `sounddevice` for non-blocking stream.
   - Implement circular buffer for waveform data.
   - Save functionality using `wave` and `numpy`.
3. **UI Components**:
   - `Header`: Title and status.
   - `Config Section`: Device dropdown, Label input, Shortcut selector.
   - `Visualizer`: Live waveform canvas.
   - `Controls`: Start/Stop button, list of recorded samples/stats.
4. **Logic**:
   - Threaded recording to keep UI responsive.
   - Mapping logic: Update `command_map.json` on label/shortcut confirmation.

## 3. Implementation Steps

- [ ] Create `data_collector.py` with skeleton structure.
- [ ] Implement `AudioRecorder` class for backend logic.
- [ ] Implement `VoiceCollectorGUI` class.
- [ ] Add styling and premium visual touches.
- [ ] Verify file handling and JSON mapping.

## 4. Verification Criteria

- [ ] GUI launches without errors.
- [ ] Microphones are listed and selectable.
- [ ] Recording saves valid `.wav` file to correct folder.
- [ ] `command_map.json` updates correctly.
- [ ] Waveform moves when sound is picked up.
