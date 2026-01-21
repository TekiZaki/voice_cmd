# Task: Transform main.py into a GUI (Pure Tkinter HUD)

## Description

Transform the current CLI-based `main.py` into a modern, information-dense HUD GUI using only standard `tkinter`.

## Plan

1. **Analysis**
   - [x] Analyze `main.py` core logic.
   - [x] Analyze `data_collector.py` GUI structure.
   - [x] Requirement gathered: Pure Tkinter, Info-dense HUD, Match History.

2. **Design (RADICAL STYLE: Pure Tkinter Cyber-HUD)**
   - **Topological Choice:** Multi-module grid layout with persistent "Monitoring" modules. Break the standard "Vertical List" by using asymmetric module sizes.
   - **Colors:** Deep Slate (`#0F172A`), Neon Blue (`#3B82F6`), Success Green (`#10B981`), Warning Gold (`#F59E0B`).
   - **Typography:** Monospaced (Consolas, Fira Code, Share Tech Mono) for that "data feed" aesthetic.
   - **Visual Elements:** Custom Canvas-drawn HUD borders, scanned-line overlays, and glow effects using color layering.

3. **Solutioning**
   - **Architecture:** `VoiceAssistantCore` (Logic) + `VoiceAssistantGUI` (UI).
   - **State Sync:** Use `threading` and a `command_queue` to safely pass data from audio inference to the UI.
   - **HUD Modules:**
     - `HeaderModule`: Title, uptime, and system status markers.
     - `StateModule`: Large "AWAKE" or "STANDBY" status with pulsing animation.
     - `VisualizerModule`: High-performance real-time oscilloscope.
     - `HistoryModule`: Scrollable log of recognized commands with confidence levels and timestamps.
     - `LogModule`: technical stream of internal events (RMS detection, model inference, action triggers).

4. **Implementation**
   - Create `voice_assistant_gui.py`.
   - Implement custom Canvas-based HUD styling (no standard ttk widgets where avoidable).
   - Port `main.py` predictive logic and action execution.
   - Add "Physical" feedback (flickers, pulses) for assistant detection.

## Verification

- [ ] Test `awake`/`standby` state transitions.
- [ ] Test command recognition and task execution.
- [ ] Test performance (ensure audio buffer doesn't lag the UI).
- [ ] UI/UX Audit: Does it look premium despite using pure Tkinter? (Check contrast, spacing, and animations).
