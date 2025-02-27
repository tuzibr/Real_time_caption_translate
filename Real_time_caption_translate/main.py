import tkinter as tk
import threading
import time
import json
import numpy as np
from tkinter import ttk, scrolledtext, filedialog
from collections import deque

import pyaudiowpatch as pyaudio

from Real_time_caption_translate.config_manager import ConfigHandler
from Real_time_caption_translate.translator import tl_api, DEEPL_LANGUAGE_TO_CODE, GOOGLE_LANGUAGES_TO_CODES

from vosk import Model, KaldiRecognizer

import sys
import os

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS,  relative_path)
    return os.path.join(os.path.abspath("."),  relative_path)

class Mainloop:
    def __init__(self, root):
        # Initialize configuration manager
        self.config_handler = ConfigHandler()
        self.current_config = self.config_handler.load_config()

        self.root = root
        self.root.title("Real-time Caption Translation")
        self.root.geometry("1200x400")
        self.root.iconbitmap(True, get_resource_path("C.ico"))
        # Save configuration when the window is closed
        self.root.protocol("WM_DELETE_WINDOW",  self.on_exit)

        # Initialize transcription state and related variables
        self.is_transcribing = False
        self.transcription_thread = None
        self.stream = None
        self.p = None
        self.rec = None
        self.tc_sentences = []  # List to store complete transcribed sentences
        self.tl_sentences = []  # List to store complete translated sentences

        self.model_dir_var = tk.StringVar(value=self.current_config["user_settings"]["model_dir"])
        self.translation_queue = deque(maxlen=2)  # Queue with a maximum length of 2
        self.queue_lock = threading.Lock()  # Thread lock for queue access

        self.source_lang = self.current_config["user_settings"]["source_lang"]
        self.target_lang = self.current_config["user_settings"]["target_lang"]

        self.engine = self.current_config["user_settings"]["engine"]
        self.current_engine_var = tk.StringVar(value=self.engine)

        # StringVars for engine-specific settings
        self.deepl_key_var = tk.StringVar(value=self.current_config["user_settings"]["deepl_key"])
        self.ollama_url_var = tk.StringVar(value=self.current_config["user_settings"]["ollama_url"])
        self.ollama_model_var = tk.StringVar(value=self.current_config["user_settings"]["ollama_model"])

        # Engine-specific language dictionaries
        self.engine_lang_dicts = {
            "Google": GOOGLE_LANGUAGES_TO_CODES,
            "DeepL": DEEPL_LANGUAGE_TO_CODE,
            "Ollama": GOOGLE_LANGUAGES_TO_CODES  # Could be empty or minimal if no selection needed
        }
        self.lang_dict = self.engine_lang_dicts.get(self.engine,
                                                    DEEPL_LANGUAGE_TO_CODE)  # Default to DeepL if engine not found

        # Monitor window properties
        self.monitor_window = None

        # Create the main interface and monitor window
        self.create_main_interface()
        self.create_monitor_window()
        self.settings_window = None

        # Audio device properties
        self.audio_devices = []  # List to store available audio devices
        self.transcribe_device = None

        # Scan audio devices on initialization
        self.scan_audio_devices()

    def scan_audio_devices(self):
        """Scan available audio input devices."""
        p = pyaudio.PyAudio()
        self.audio_devices = []

        try:
            # Get WASAPI information
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            if not default_speakers["isLoopbackDevice"]:
                for loopback in p.get_loopback_device_info_generator():
                    """
                    Try to find loopback device with same name(and [Loopback suffix]).
                    Unfortunately, this is the most adequate way at the moment.
                    """
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break

            self.audio_devices.append({
                "name": f"[Speaker] {default_speakers['name']}",
                "index": default_speakers["index"],
                "channels": default_speakers["maxInputChannels"],
                "rate": int(default_speakers["defaultSampleRate"])
            })

            default_microphone = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
            self.audio_devices.append({
                "name": f"[Microphone] {default_microphone['name']}",
                "index": default_microphone["index"],
                "channels": default_microphone["maxInputChannels"],
                "rate": int(default_microphone["defaultSampleRate"])
            })

            self.transcribe_device = self.audio_devices[0] if self.audio_devices else None

        except OSError as e:
            print(f"Error scanning audio devices: {e}")


    def create_main_interface(self):
        """Create the main user interface."""
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        settings_btn = ttk.Button(toolbar, text="⚙️ Settings", command=self.open_settings)
        settings_btn.pack(side=tk.LEFT)

        # Monitor toggle button
        self.monitor_btn = ttk.Button(toolbar, text="📺 Hide", command=self.toggle_monitor)
        self.monitor_btn.pack(side=tk.LEFT, padx=5)

        # Source language selector
        ttk.Label(toolbar, text="Source Language:").pack(side=tk.LEFT, padx=5)
        self.source_lang_selector = ttk.Combobox(toolbar, values=list(self.lang_dict.keys()))
        self.source_lang_selector.pack(side=tk.LEFT, padx=5)
        self.source_lang_selector.set(self.source_lang)

        # Target language selector
        ttk.Label(toolbar, text="Target Language:").pack(side=tk.LEFT, padx=5)
        self.target_lang_selector = ttk.Combobox(toolbar, values=list(self.lang_dict.keys()))
        self.target_lang_selector.pack(side=tk.LEFT, padx=5)
        self.target_lang_selector.set(self.target_lang)

        # Start/Stop button
        self.start_stop_btn = ttk.Button(toolbar, text="Start", command=self.toggle_transcription)
        self.start_stop_btn.pack(side=tk.RIGHT, padx=5)

        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Transcription text area
        self.source_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=('Arial', 14),
            padx=5,
            pady=5,
            bg='#f0f0f0',
            state="disabled"
        )
        self.source_text.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        # Translation text area
        self.translated_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=('Arial', 14),
            padx=5,
            pady=5,
            bg='#f0f0f0',
            state="disabled"
        )
        self.translated_text.grid(row=1, column=1, sticky="nsew")

    def create_monitor_window(self):
        """Create a borderless, interactive monitor window."""
        self.monitor_window = tk.Toplevel(self.root)
        self.monitor_window.overrideredirect(True)
        pos = self.current_config["user_settings"]["monitor_position"]
        self.monitor_window.geometry(f"1000x200+{pos[0]}+{pos[1]}")
        self.monitor_window.attributes('-topmost', True)
        self.monitor_window.attributes('-alpha', 1)
        self.monitor_window.config(borderwidth=0, relief='groove')

        # Bind window drag events
        self.monitor_window.bind("<B1-Motion>", self.drag_monitor)
        self.monitor_window.bind("<Button-1>", self.start_drag)

        # Split pane for transcription and translation
        self.monitor_pane = ttk.PanedWindow(self.monitor_window, orient=tk.VERTICAL)
        self.monitor_pane.pack(fill=tk.BOTH, expand=True)

        # Transcription monitor area
        self.partial_transcript = tk.Text(
            self.monitor_pane,
            wrap=tk.WORD,
            font=('Arial', 12),
            bg='#FAFAFA',
            padx=10,
            pady=10,
            relief='flat'
        )
        self.monitor_pane.add(self.partial_transcript, weight=1)

        # Translation monitor area
        self.partial_translation = tk.Text(
            self.monitor_pane,
            wrap=tk.WORD,
            font=('Arial', 12),
            bg='#F0F8FF',
            padx=10,
            pady=10,
            relief='flat'
        )
        self.monitor_pane.add(self.partial_translation, weight=1)

        # Resize handle
        self.resize_handle = ttk.Sizegrip(self.monitor_window)
        self.resize_handle.place(relx=1.0, rely=1.0, anchor='se')

    def drag_monitor(self, event):
        """Handle dragging of the monitor window."""
        x = self.monitor_window.winfo_x() + (event.x - self.drag_data["x"])
        y = self.monitor_window.winfo_y() + (event.y - self.drag_data["y"])
        self.monitor_window.geometry(f"+{x}+{y}")

    def start_drag(self, event):
        """Record the starting point for dragging."""
        self.drag_data = {"x": event.x, "y": event.y}

    def toggle_monitor(self):
        """Toggle the visibility of the monitor window."""
        if self.monitor_window.winfo_viewable():
            self.monitor_window.withdraw()
            self.monitor_btn.config(text="📺  Show")
        else:
            self.monitor_window.deiconify()
            self.monitor_btn.config(text="📺  Hide")

    def toggle_transcription(self):
        """Toggle the transcription state."""
        if not self.is_transcribing:
            self.start_transcription()
        else:
            self.stop_transcription()

    def convert_to_mono(self, data, channels):
        """
        Convert multi-channel audio data to mono using NumPy for better performance.
        :param data: Raw audio data in bytes
        :param channels: Number of audio channels
        :return: Mono audio data in bytes
        """
        if channels == 1:
            return data

        samples = np.frombuffer(data, dtype='<i2')
        num_frames = len(samples) // channels
        samples_reshaped = samples.reshape(num_frames, channels)
        mono_samples = np.sum(samples_reshaped, axis=1, dtype=np.int32) // channels
        return mono_samples.astype('<i2').tobytes()

    def start_transcription(self):
        """Start the transcription process."""
        if self.is_transcribing:
            return

        with self.queue_lock:
            self.translation_queue.clear()
        self.tc_sentences.clear()
        self.tl_sentences.clear()

        self.is_transcribing = True
        self.start_stop_btn.config(text="Stop")
        self.source_text.config(state="normal")
        self.source_text.delete(1.0, tk.END)
        self.source_text.config(state="disabled")
        self.source_text.tag_configure("partial", foreground="gray")

        self.translated_text.config(state="normal")
        self.translated_text.delete(1.0, tk.END)
        self.translated_text.config(state="disabled")
        self.translated_text.tag_configure("partial", foreground="gray")

        # Initialize audio stream and Vosk recognizer
        CHUNK = 4096
        self.p = pyaudio.PyAudio()

        if 'Microphone' in self.transcribe_device['name']:
            self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=self.transcribe_device["channels"],
                        rate=self.transcribe_device["rate"],
                        frames_per_buffer=CHUNK,
                        input=True,
                        input_device_index=self.transcribe_device["index"],
                        )
        else:
            self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=self.transcribe_device["channels"],
                        rate=self.transcribe_device["rate"],
                        frames_per_buffer=CHUNK,
                        input=True,
                        input_device_index=self.transcribe_device["index"],
                        )

        model = Model(self.model_dir_var.get())
        self.rec = KaldiRecognizer(model, self.transcribe_device["rate"])

        # Start transcription and translation threads
        self.transcription_thread = threading.Thread(target=self.transcription_loop, daemon=True)
        self.transcription_thread.start()
        self.translation_thread = threading.Thread(target=self.translation_loop, daemon=True)
        self.translation_thread.start()

    def stop_transcription(self):
        """Stop the transcription process."""
        if not self.is_transcribing:
            return

        self.is_transcribing = False

        # Wait for threads to finish with a timeout
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2)
        if self.translation_thread and self.translation_thread.is_alive():
            self.translation_thread.join(timeout=2)

        with self.queue_lock:
            self.translation_queue.clear()

        # Clean up audio resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

        self.stream = None
        self.p = None
        self.rec = None

        print("Transcription stopped.")
        self.start_stop_btn.config(text="Start")

    def transcription_loop(self):
        """Main loop for audio transcription."""
        while self.is_transcribing and self.rec is not None:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                data = self.convert_to_mono(data, self.transcribe_device["channels"])
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    if text:
                        self.tc_sentences.append(text)
                        self.root.after(0, self.update_source_text, text, True)

                        tl_task = {"text": text, "flag": True}
                        self.translation_queue.append(tl_task)

                else:
                    if self.rec:
                        partial = json.loads(self.rec.PartialResult())
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            self.root.after(0, self.update_source_text, partial_text, False)

                            if not self.translation_queue:
                                tl_task = {"text": partial_text, "flag": False}
                                self.translation_queue.append(tl_task)

            except Exception as e:
                print(f"Transcription error: {e}")
                break

    def translation_loop(self):
        while self.is_transcribing:
            task = None
            with self.queue_lock:
                if self.translation_queue:
                    task = self.translation_queue.popleft()

            if task:
                try:
                    engine = self.current_engine_var.get()
                    kwargs = {}
                    if engine != "Ollama":
                        source_lang_code = self.lang_dict[self.source_lang_selector.get()]
                        target_lang_code = self.lang_dict[self.target_lang_selector.get()]
                        kwargs["lang_source"] = source_lang_code
                        kwargs["lang_target"] = target_lang_code
                    if engine == "DeepL":
                        kwargs["api_key"] = self.deepl_key_var.get()
                    elif engine == "Ollama":
                        kwargs["url"] = self.ollama_url_var.get()
                        kwargs["model"] = self.ollama_model_var.get()
                        kwargs["lang_target"] = self.target_lang_selector.get()

                    if task['flag']:
                        translated = tl_api(engine=engine, text=task['text'], **kwargs)
                        self.tl_sentences.append(translated)
                        self.root.after(0, self.update_translated_text, translated, True)
                    else:
                        translated = tl_api(engine=engine, text=task['text'], **kwargs)
                        self.root.after(0, self.update_translated_text, translated, False)
                except Exception as e:
                    print(f"Translation error: {e}")
            else:
                time.sleep(0.1)

    def update_source_text(self, text, is_complete):
        """Update the transcription text area."""
        self.source_text.config(state="normal")

        if is_complete:
            self._clear_partial_text()
            self.source_text.insert("end", text + "\n")
        else:
            self._clear_partial_text()
            self.source_text.insert("end", text + " ", "partial")
            self._update_monitor_text(self.partial_transcript,text + " ")

        self.source_text.config(state="disabled")
        self.source_text.bindtags((self.source_text, self.root, "all"))
        self.source_text.see(tk.END)

    def update_translated_text(self, text, is_complete):
        """Main loop for translating transcribed text."""
        self.translated_text.config(state="normal")

        if is_complete:
            self._clear_translated_partial_text()
            self.translated_text.insert("end", text + "\n")
        else:
            self._clear_translated_partial_text()
            self.translated_text.insert("end", text + " ", "partial")
            self._update_monitor_text(self.partial_translation,text + " ")

        self.translated_text.config(state="disabled")
        self.translated_text.bindtags((self.translated_text, self.root, "all"))
        self.translated_text.see(tk.END)

    def _update_monitor_text(self, widget, text):
        """Update text in the monitor window."""
        widget.config(state='normal')
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state='disabled')
        widget.see(tk.END)

    def _clear_partial_text(self):
        """Safely clear partial transcription text."""
        try:
            start_idx = self.source_text.tag_ranges("partial")[0]
            end_idx = self.source_text.tag_ranges("partial")[1]
            self.source_text.delete(start_idx, end_idx)
        except IndexError:
            pass

    def _clear_translated_partial_text(self):
        """Safely clear partial translation text."""
        try:
            start_idx = self.translated_text.tag_ranges("partial")[0]
            end_idx = self.translated_text.tag_ranges("partial")[1]
            self.translated_text.delete(start_idx, end_idx)
        except IndexError:
            pass

    def open_settings(self):
        """Open the settings window."""
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = tk.Toplevel(self.root)
            self.settings_window.title("Settings")
            self.settings_window.geometry("600x300")

            notebook = ttk.Notebook(self.settings_window)

            audio_tab = ttk.Frame(notebook)
            self.create_audio_settings(audio_tab)
            notebook.add(audio_tab, text="Audio Settings")

            trans_tab = ttk.Frame(notebook)
            self.create_translation_settings(trans_tab)
            notebook.add(trans_tab, text="Translation Settings")

            notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def create_audio_settings(self, parent):
        """Create the audio settings interface."""
        ttk.Label(parent, text="Transcription Device:").grid(row=0, column=0, sticky=tk.W)
        device_names = [f"{dev['name']} ({dev['rate']}Hz)" for dev in self.audio_devices]
        self.input_devices = ttk.Combobox(parent, values=device_names, width=50)
        self.input_devices.grid(row=0, column=1, sticky=tk.EW)

        self.transcribe_device = self.audio_devices[0] if self.audio_devices else None
        self.input_devices.current(0)
        self.input_devices.bind("<<ComboboxSelected>>", self.on_device_select)

        ttk.Label(parent, text="Recognition Model Path:").grid(row=1, column=0, sticky=tk.W)

        path_frame = ttk.Frame(parent)
        path_frame.grid(row=1, column=1, sticky=tk.EW)

        entry = ttk.Entry(path_frame, textvariable=self.model_dir_var, width=40)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(path_frame, text="Browse...", width=8, command=self.browse_model_dir)
        browse_btn.pack(side=tk.RIGHT, padx=5)

    def browse_model_dir(self):
        """Open a directory selection dialog for the model path."""
        selected_dir = filedialog.askdirectory(title="Select Speech Model Directory",
                                               initialdir=self.model_dir_var.get())
        if selected_dir:
            self.model_dir_var.set(selected_dir)

    def on_device_select(self, event):
        """Handle audio device selection."""
        selected_idx = self.input_devices.current()
        if selected_idx >= 0 and selected_idx < len(self.audio_devices):
            self.transcribe_device = self.audio_devices[selected_idx]
            print(f"Selected device{self.transcribe_device['name']}")

    def create_translation_settings(self, parent):
        """Create the translation settings interface."""
        ttk.Label(parent, text="Translation Engine:").grid(row=0, column=0, sticky=tk.W)
        self.trans_engine = ttk.Combobox(parent, values=["Google", "DeepL", "Ollama"], textvariable=self.current_engine_var)
        self.trans_engine.grid(row=0, column=1, sticky=tk.EW)
        self.trans_engine.bind("<<ComboboxSelected>>", self.on_engine_select)

        # Frame for engine-specific settings
        self.engine_settings_frame = ttk.Frame(parent)
        self.engine_settings_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        # Initial update of settings frame
        self.update_engine_settings()

    def on_engine_select(self, event):
        """Handle engine selection change."""
        self.update_engine_settings()
        self.update_language_selectors()

    def update_engine_settings(self):
        """Update the engine-specific settings UI based on selected engine."""
        engine = self.current_engine_var.get()
        # Clear existing widgets
        for widget in self.engine_settings_frame.winfo_children():
            widget.destroy()

        if engine == "DeepL":
            ttk.Label(self.engine_settings_frame, text="DeepL API Key:").grid(row=0, column=0, sticky=tk.W)
            self.deepl_key_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.deepl_key_var, width=35)
            self.deepl_key_entry.grid(row=0, column=1, sticky=tk.EW)
        elif engine == "Ollama":
            ttk.Label(self.engine_settings_frame, text="Ollama URL:").grid(row=0, column=0, sticky=tk.W)
            self.ollama_url_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.ollama_url_var, width=35)
            self.ollama_url_entry.grid(row=0, column=1, sticky=tk.EW)
            ttk.Label(self.engine_settings_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W)
            self.ollama_model_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.ollama_model_var, width=35)
            self.ollama_model_entry.grid(row=1, column=1, sticky=tk.EW)
        # For Google, no additional settings

    def update_language_selectors(self):
        """Update language selectors based on selected engine."""
        engine = self.trans_engine.get()

        self.source_lang_selector.config(state="normal")
        self.target_lang_selector.config(state="normal")
        if engine in self.engine_lang_dicts:
            self.lang_dict = self.engine_lang_dicts[engine]
            languages = list(self.lang_dict.keys())
            self.source_lang_selector['values'] = languages
            self.target_lang_selector['values'] = languages
            # Set to current languages if available, else first option
            self.source_lang_selector.set(
                self.source_lang if self.source_lang in languages else languages[0] if languages else "")
            self.target_lang_selector.set(
                self.target_lang if self.target_lang in languages else languages[0] if languages else "")

    def on_exit(self):
        """Handle window close and save configuration."""
        current_settings = {
            "user_settings": {
                "engine": self.current_engine_var.get(),
                "source_lang": self.source_lang_selector.get(),
                "target_lang": self.target_lang_selector.get(),
                "model_dir": self.model_dir_var.get(),
                "transcribe_device_index": self.audio_devices.index(
                    self.transcribe_device) if self.transcribe_device else 0,
                "monitor_position": [
                    self.monitor_window.winfo_x(),
                    self.monitor_window.winfo_y()
                ],
                "deepl_key": self.deepl_key_var.get(),
                "ollama_url": self.ollama_url_var.get(),
                "ollama_model": self.ollama_model_var.get()
            }
        }
        self.config_handler.save_config(current_settings)
        self.root.destroy()

def main():
    root = tk.Tk()
    Real_time_caption_translate = Mainloop(root)
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = Mainloop(root)
    root.mainloop()