import tkinter as tk
import threading
import time
import json
import numpy as np
from tkinter import ttk, scrolledtext, filedialog, messagebox
from collections import deque
from sys import platform
from copy import deepcopy
from librosa import resample
from multiprocessing import Process, Queue

if platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio


import logging

from Real_time_caption_translate.config_manager import ConfigHandler
from Real_time_caption_translate.translator import tl_api, DEEPL_LANGUAGE_TO_CODE, GOOGLE_LANGUAGES_TO_CODES
from Real_time_caption_translate.hotwords import correct_sentence

import sys
import os

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS,  relative_path)
    return os.path.join(os.path.abspath("."),  relative_path)


def transcription_process(data_queue, result_queue, model_dir, transcribe_device):
    from Real_time_caption_translate.whisper_online import FasterWhisperASR, OnlineASRProcessor
    from librosa import resample
    import numpy as np

    asr = FasterWhisperASR("en", model_dir=model_dir)
    asr.use_vad()
    rec = OnlineASRProcessor(asr)

    while True:
        if not data_queue.empty():
            data = data_queue.get()

            if True:

                audio_int16 = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                mono = (audio_float[0::2] + audio_float[1::2]) / 2.0 if transcribe_device["channels"] > 1 else audio_float
                resampled = resample(mono, orig_sr=transcribe_device["rate"], target_sr=16000)
                data = resampled.astype(np.float32)


                rec.insert_audio_chunk(data)
                stream_text = rec.process_iter()
                if stream_text[2] != "":
                    result_queue.put(stream_text)
        else:
            time.sleep(0.1)

class Mainloop:
    def __init__(self, root):
        # Initialize configuration manager
        self.config_handler = ConfigHandler()
        self.current_config = self.config_handler.load_config()

        self.root = root
        self.root.title("Real-time Caption Translation")
        self.root.geometry("1200x400")

        if platform == "win32":
            self.root.iconbitmap(True, get_resource_path("C.ico"))
        else:
            self.root.iconphoto(True, tk.PhotoImage(file=get_resource_path("C.png")))

        # Save configuration when the window is closed
        self.root.protocol("WM_DELETE_WINDOW",  self.on_exit)

        # Initialize transcription state and related variables
        self.is_transcribing = False
        self.transcription_thread = None
        self.stream = None
        self.p = None
        self.rec = None
        self.chuck = 48000
        self.tc_sentences = []  # List to store complete transcribed sentences
        self.tl_sentences = []  # List to store complete translated sentences
        self.stream_text = ""

        self.model_dir_var = tk.StringVar(value=self.current_config["user_settings"]["model_dir"])
        self.translation_queue = Queue(maxsize=2)


        self.data_queue = Queue()
        self.result_queue = Queue()
        self.transcription_process = None

        self.source_lang = self.current_config["user_settings"]["source_lang"]
        self.target_lang = self.current_config["user_settings"]["target_lang"]

        self.engine = self.current_config["user_settings"]["engine"]
        self.current_engine_var = tk.StringVar(value=self.engine)

        self.translate_when_sentence_finishes = self.current_config["user_settings"]["translate_when_sentence_finishes"]

        # StringVars for engine-specific settings
        self.deepl_key_var = tk.StringVar(value=self.current_config["user_settings"]["deepl_key"])
        self.deepseek_key_var = tk.StringVar(value=self.current_config["user_settings"]["deepseek_key"])
        self.ollama_url_var = tk.StringVar(value=self.current_config["user_settings"]["ollama_url"])
        self.ollama_model_var = tk.StringVar(value=self.current_config["user_settings"]["ollama_model"])
        self.openai_url_var = tk.StringVar(value=self.current_config["user_settings"]["openai_url"])
        self.openai_key_var = tk.StringVar(value=self.current_config["user_settings"]["openai_key"])
        self.openai_model_var = tk.StringVar(value=self.current_config["user_settings"]["openai_model"])

        # Engine-specific language dictionaries
        self.engine_lang_dicts = {
            "Google": GOOGLE_LANGUAGES_TO_CODES,
            "DeepL": DEEPL_LANGUAGE_TO_CODE,
            "DeepSeek": GOOGLE_LANGUAGES_TO_CODES,
            "Ollama": GOOGLE_LANGUAGES_TO_CODES,
            "OpenAI": GOOGLE_LANGUAGES_TO_CODES
        }
        self.lang_dict = self.engine_lang_dicts.get(self.engine,
                                                    DEEPL_LANGUAGE_TO_CODE)  # Default to DeepL if engine not found

        self.hotwords = self.current_config["user_settings"].get("hotwords")
        self.hotwords_beta = False

        # Audio device properties
        self.audio_devices = []  # List to store available audio devices
        self.transcribe_device = None

        # Scan audio devices on initialization
        self.scan_audio_devices()

        self.is_monitor_visible = False

        # Monitor window properties
        self.monitor_window = None

        # Create the main interface and monitor window
        self.create_main_interface()
        self.create_monitor_window()
        self.settings_window = None

    def create_main_interface(self):
        """Create the main user interface."""
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        settings_btn = ttk.Button(toolbar, text="⚙️ Settings", command=self.open_settings)
        settings_btn.pack(side=tk.LEFT)

        # Monitor toggle button
        self.monitor_btn = ttk.Button(toolbar, text="📺 Show", command=self.toggle_monitor)
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

        self.hotwords_var = tk.BooleanVar(value=self.hotwords_beta)

        self.hotwords_cbtn = ttk.Checkbutton(
            toolbar,
            text="Hotwords",
            variable=self.hotwords_var,
            command=self.toggle_hotwords_beta,
            style="Toggle.TCheckbutton"
        )
        self.hotwords_cbtn.pack(side=tk.RIGHT, padx=5)

        style = ttk.Style()
        style.configure("Toggle.TCheckbutton",
                        font=('Microsoft YaHei', 9),
                        foreground="#444",
                        indicatormargin=4,
                        indicatorsize=16)


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
            font=('TkDefaultFont', 14),
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
            font=('TkDefaultFont', 14),
            padx=5,
            pady=5,
            bg='#f0f0f0',
            state="disabled"
        )
        self.translated_text.grid(row=1, column=1, sticky="nsew")

        self.create_context_menu(self.source_text)
        self.create_context_menu(self.translated_text)

    def create_context_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="copy", command=lambda: self.copy_text(widget))
        widget.bind("<Button-3>", lambda event: self.show_context_menu(event, menu))

    def show_context_menu(self, event, menu):
        menu.tk_popup(event.x_root, event.y_root)

    def copy_text(self, widget):
        try:
            selected_text = widget.selection_get()
            widget.clipboard_clear()
            widget.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def create_monitor_window(self):
        """Create a borderless, interactive monitor window."""
        self.monitor_window = tk.Toplevel(self.root)
        self.monitor_window.overrideredirect(True)
        pos = self.current_config["user_settings"]["monitor_position"]
        self.monitor_window.geometry(f"1000x200+{pos[0]}+{pos[1]}")
        self.monitor_window.attributes('-topmost', True)
        self.monitor_window.attributes('-alpha', 0.9)
        self.monitor_window.config(bg='#000000')

        # Bind window drag events
        self.monitor_window.bind("<B1-Motion>", self.drag_monitor)
        self.monitor_window.bind("<Button-1>", self.start_drag)

        style = ttk.Style()
        style.configure("Black.TPanedwindow", background="#000000", sashwidth=10, sashrelief="flat")

        # Split pane for transcription and translation
        self.monitor_pane = ttk.PanedWindow(self.monitor_window, orient=tk.VERTICAL, style="Black.TPanedwindow")
        self.monitor_pane.pack(fill=tk.BOTH, expand=True)

        # Transcription monitor area
        self.partial_transcript = tk.Text(
            self.monitor_pane,
            wrap=tk.WORD,
            font=('TkDefaultFont', 14),
            bg='#000000',
            fg='#FFFFFF',
            padx=10,
            pady=10,
            relief='flat'
        )
        self.partial_transcript.config(state='disabled')
        self.monitor_pane.add(self.partial_transcript, weight=1)

        # Translation monitor area
        self.partial_translation = tk.Text(
            self.monitor_pane,
            wrap=tk.WORD,
            font=('TkDefaultFont', 14),
            bg='#000000',
            fg='#FFFFFF',
            padx=10,
            pady=10,
            relief='flat'
        )
        self.partial_translation.config(state='disabled')
        self.monitor_pane.add(self.partial_translation, weight=1)

        # Resize handle
        self.resize_handle = ttk.Sizegrip(self.monitor_window)
        self.resize_handle.place(relx=1.0, rely=1.0, anchor='se')

        self.source_text.bind("<Button-1>", lambda e: self.source_text.focus_set())
        self.translated_text.bind("<Button-1>", lambda e: self.translated_text.focus_set())

        self.monitor_window.withdraw()

    def drag_monitor(self, event):
        """Handle dragging of the monitor window."""
        x = self.monitor_window.winfo_x() + (event.x - self.drag_data["x"])
        y = self.monitor_window.winfo_y() + (event.y - self.drag_data["y"])
        self.monitor_window.geometry(f"+{x}+{y}")

    def start_drag(self, event):
        """Record the starting point for dragging."""
        self.drag_data = {"x": event.x, "y": event.y}

    def scan_audio_devices(self):
        """Scan available audio input devices."""
        self.audio_devices = []

        if platform == "win32":
            try:
                p = pyaudio.PyAudio()
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

        else:
            # Use standard PyAudio for macOS and other platforms
            p = pyaudio.PyAudio()
            try:
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev["maxInputChannels"] > 0:  # Only input devices
                        self.audio_devices.append({
                            "name": f"[Microphone] {dev['name']}",
                            "index": dev["index"],
                            "channels": dev["maxInputChannels"],
                            "rate": int(dev["defaultSampleRate"])
                        })
                if not self.audio_devices:
                    print("No input devices found on this system.")

                self.transcribe_device = self.audio_devices[0] if self.audio_devices else None
            except Exception as e:
                print(f"Error scanning audio devices: {e}")

    def toggle_hotwords_beta(self):
        """Toggle the Hotwords Beta"""
        self.hotwords_beta = not self.hotwords_beta
        self.hotwords_var.set(self.hotwords_beta)

    def toggle_monitor(self):
        """Toggle the visibility of the monitor window."""
        if self.is_monitor_visible:
            self.monitor_window.withdraw()
            self.monitor_btn.config(text="📺 Show")
        else:
            # 如果当前隐藏，则显示
            self.monitor_window.deiconify()
            self.monitor_btn.config(text="📺 Hide")
        self.is_monitor_visible = not self.is_monitor_visible

    def toggle_transcription(self):
        """Toggle the transcription state."""
        if not self.is_transcribing:
            self.start_transcription()
        else:
            self.stop_transcription()

    def start_transcription(self):
        if self.is_transcribing:
            return

        self.tc_sentences.clear()
        self.tl_sentences.clear()
        self.stream_text = ""
        self.chuck = self.transcribe_device["rate"]

        self.source_text.config(state="normal")
        self.source_text.delete(1.0, tk.END)
        self.source_text.config(state="disabled")
        self.source_text.tag_configure("partial", foreground="gray")

        self.translated_text.config(state="normal")
        self.translated_text.delete(1.0, tk.END)
        self.translated_text.config(state="disabled")
        self.translated_text.tag_configure("partial", foreground="gray")

        self.p = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            self.data_queue.put(in_data)
            return (in_data, pyaudio.paContinue)

        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=self.transcribe_device["channels"],
                                      rate=self.transcribe_device["rate"],
                                      frames_per_buffer=self.chuck,
                                      input=True,
                                      input_device_index=self.transcribe_device["index"],
                                      stream_callback=callback
                                      )
        except Exception as e:
            logging.error(f"Error initializing audio stream: {e}")
            messagebox.showerror("Error", f"Error initializing audio stream: {e}")
            return

        self.data_queue = Queue()
        self.result_queue = Queue()

        self.transcription_process = Process(target=transcription_process,
                                             args=(self.data_queue, self.result_queue, self.model_dir_var.get(),
                                                   self.transcribe_device))
        self.transcription_process.start()

        self.start_stop_btn.config(text="Stop")
        self.is_transcribing = True
        logging.info("Starting transcription")

        self.rec_hot_words = None
        self.rec_hot_words = deepcopy(self.current_config["user_settings"].get("hotwords"))


        def process_results():
            while not self.result_queue.empty():
                stream_text = self.result_queue.get()
                if stream_text[2] != "":
                    self.stream_text += stream_text[2]
                    text, self.stream_text = self.process_string(self.stream_text)
                    if text:
                        if self.hotwords_beta:
                            text = correct_sentence(text, self.rec_hot_words)
                        self.root.after(10, self.update_source_text, text, True)
                        tl_task = {"text": text, "flag": True}
                        self.translation_queue.put(tl_task)
                    else:
                        if not self.translate_when_sentence_finishes:
                            tl_task = {"text": self.stream_text, "flag": False}
                            try:
                                self.translation_queue.put_nowait(tl_task)
                            except :
                                pass

                    self.root.after(10, self.update_source_text, self.stream_text, False)

            if self.is_transcribing:
                self.root.after(10, process_results)

        self.root.after(10, process_results)

        self.translation_thread = threading.Thread(target=self.translation_loop, daemon=True)
        self.translation_thread.start()

    def stop_transcription(self):
        if not self.is_transcribing:
            return

        while not self.data_queue.empty():
            self.data_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
        while not self.translation_queue.empty():
            self.translation_queue.get()

        self.is_transcribing = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        if self.transcription_process:
            self.transcription_process.terminate()
            self.transcription_process.join()
            self.transcription_process = None

        self.stream = None
        self.p = None

        self.start_stop_btn.config(text="Start")
        logging.info("Transcription stopped.")

    def process_string(self, s):
        parts = s.split('.')
        count = len(parts) - 1
        if count <= 1:
            return False, s
        else:
            first_part = parts[0] + '.' + parts[1] + '.'
            second_part = ''.join(parts[2:])
            return first_part, second_part


    def translation_loop(self):
        while self.is_transcribing:
            if not self.translation_queue.empty():

                task = self.translation_queue.get(timeout=0.1)

                try:
                    engine = self.current_engine_var.get()
                    kwargs = {}

                    if engine == "Google":
                        source_lang_code = self.lang_dict[self.source_lang_selector.get()]
                        target_lang_code = self.lang_dict[self.target_lang_selector.get()]
                        kwargs["lang_source"] = source_lang_code
                        kwargs["lang_target"] = target_lang_code
                    elif engine == "DeepL":
                        kwargs["api_key"] = self.deepl_key_var.get()
                        source_lang_code = self.lang_dict[self.source_lang_selector.get()]
                        target_lang_code = self.lang_dict[self.target_lang_selector.get()]
                        kwargs["lang_source"] = source_lang_code
                        kwargs["lang_target"] = target_lang_code
                    elif engine == "Ollama":
                        kwargs["url"] = self.ollama_url_var.get()
                        kwargs["model"] = self.ollama_model_var.get()
                        kwargs["lang_source"] = self.source_lang_selector.get()
                        kwargs["lang_target"] = self.target_lang_selector.get()
                    elif engine == "DeepSeek":
                        kwargs["api_key"] = self.deepseek_key_var.get()
                        kwargs["lang_source"] = self.source_lang_selector.get()
                        kwargs["lang_target"] = self.target_lang_selector.get()
                    elif engine == "OpenAI":
                        kwargs["url"] = self.openai_url_var.get()
                        kwargs["model"] = self.openai_model_var.get()
                        kwargs["api_key"] = self.openai_key_var.get()
                        kwargs["lang_source"] = self.source_lang_selector.get()
                        kwargs["lang_target"] = self.target_lang_selector.get()

                    if task['flag']:
                        translated = tl_api(engine=engine, text=task['text'], **kwargs)
                        self.tl_sentences.append(translated)
                        self.root.after(10, self.update_translated_text, translated, True)
                    else:
                        translated = tl_api(engine=engine, text=task['text'], **kwargs)
                        self.root.after(10, self.update_translated_text, translated, False)
                except Exception as e:
                    print(f"Translation error: {e}")
            else:
                time.sleep(0.1)

    def update_source_text(self, text, is_complete):
        """Update the transcription text area."""
        self.source_text.config(state="normal")

        first, last = self.source_text.yview()
        was_at_bottom = (last == 1.0)

        if is_complete:
            self._clear_partial_text()
            self.source_text.insert("end", text + "\n")
        else:
            self._clear_partial_text()
            self.source_text.insert("end", text + " ", "partial")
            self._update_monitor_text(self.partial_transcript,text + " ")

        if was_at_bottom:
            self.source_text.see(tk.END)

    def update_translated_text(self, text, is_complete):
        """Main loop for translating transcribed text."""
        self.translated_text.config(state="normal")

        first, last = self.translated_text.yview()
        was_at_bottom = (last == 1.0)

        if is_complete:
            self._clear_translated_partial_text()
            self.translated_text.insert("end", text + "\n")
        else:
            self._clear_translated_partial_text()
            self.translated_text.insert("end", text + " ", "partial")
            self._update_monitor_text(self.partial_translation,text + " ")

        if was_at_bottom:
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

            hotwords_tab = ttk.Frame(notebook)
            self.create_hotwords_settings(hotwords_tab)
            notebook.add(hotwords_tab, text="Hotwords Settings")

            save_frame = ttk.Frame(self.settings_window)
            save_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

            save_all_btn = ttk.Button(save_frame,
                                      text="Save Settings",
                                      command=lambda: [
                                          self.save_all_config(),
                                          self.settings_window.destroy()  # 新增关闭操作
                                      ])
            save_all_btn.pack(side=tk.RIGHT, anchor=tk.E)

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

        entry = ttk.Entry(path_frame, textvariable=self.model_dir_var, width=50)
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
        # frame
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=5)

        ttk.Label(header_frame, text="Translation Engine:").grid(row=0, column=0, sticky=tk.W)
        self.trans_engine = ttk.Combobox(header_frame, values=["Google", "DeepL", "Ollama", "DeepSeek", "OpenAI"],
                                         textvariable=self.current_engine_var)
        self.trans_engine.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10))
        self.trans_engine.bind("<<ComboboxSelected>>", self.on_engine_select)

        # translate_when_sentence_finishes
        self.translate_when_sentence_finishes_var = tk.BooleanVar(value=self.translate_when_sentence_finishes)
        self.translate_when_sentence_finishes_cbtn = ttk.Checkbutton(
            header_frame,
            text="translate_when_sentence_finishes",
            variable=self.translate_when_sentence_finishes_var,
            command=self.toggle_translate_when_sentence_finishes,
            style="Toggle.TCheckbutton"
        )
        self.translate_when_sentence_finishes_cbtn.grid(row=0, column=2, padx=5, sticky=tk.E)

        header_frame.columnconfigure(1, weight=1)

        self.engine_settings_frame = ttk.Frame(parent)
        self.engine_settings_frame.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)

        self.update_engine_settings()

    def toggle_translate_when_sentence_finishes(self):
        """Toggle the translation when sentence finishes checkbox."""
        self.translate_when_sentence_finishes = not self.translate_when_sentence_finishes
        self.translate_when_sentence_finishes_var.set(self.translate_when_sentence_finishes)

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
            self.deepl_key_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.deepl_key_var, width=40)
            self.deepl_key_entry.grid(row=0, column=1, sticky=tk.EW)
        elif engine == "Ollama":
            ttk.Label(self.engine_settings_frame, text="Ollama URL:").grid(row=0, column=0, sticky=tk.W)
            self.ollama_url_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.ollama_url_var, width=40)
            self.ollama_url_entry.grid(row=0, column=1, sticky=tk.EW)
            ttk.Label(self.engine_settings_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W)
            self.ollama_model_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.ollama_model_var, width=40)
            self.ollama_model_entry.grid(row=1, column=1, sticky=tk.EW)
        elif engine == "DeepSeek":
            ttk.Label(self.engine_settings_frame, text="DeepSeek API Key:").grid(row=0, column=0, sticky=tk.W)
            self.deepseek_key_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.deepseek_key_var, width=40)
            self.deepseek_key_entry.grid(row=0, column=1, sticky=tk.EW)
        elif engine == "OpenAI":
            ttk.Label(self.engine_settings_frame, text="OpenAI URL:").grid(row=0, column=0, sticky=tk.W)
            self.openai_url_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.openai_url_var, width=40)
            self.openai_url_entry.grid(row=0, column=1, sticky=tk.EW)

            ttk.Label(self.engine_settings_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W)
            self.openai_model_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.openai_model_var,width=40)
            self.openai_model_entry.grid(row=1, column=1, sticky=tk.EW)

            ttk.Label(self.engine_settings_frame, text="OpenAI API Key:").grid(row=2, column=0, sticky=tk.W)
            self.openai_key_entry = ttk.Entry(self.engine_settings_frame, textvariable=self.openai_key_var, width=40)
            self.openai_key_entry.grid(row=2, column=1, sticky=tk.EW)
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

    def create_hotwords_settings(self, parent):
        """Create the hotwords settings interface"""
        # Add a hint label
        ttk.Label(parent,
                  text="The hotwords list allows you to improve the recognition accuracy of specific words during speech recognition.").grid(
            row=0, column=0,
            columnspan=2,
            sticky=tk.W, pady=2)

        # Hotwords input area
        ttk.Label(parent, text="Enter hotwords (one per line, Use lowercase letters):").grid(row=1, column=0, sticky=tk.W)
        self.hotwords_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=40, height=10)
        self.hotwords_text.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        self.hotwords_text.insert("end", "\n".join(self.hotwords))  # Display existing hotwords

        # Make the input box resize with the window
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)


    def on_exit(self):
        """Handle window close"""
        while self.is_transcribing:
            self.stop_transcription()
            time.sleep(0.1)

        self.save_all_config(False)
        self.root.destroy()

    def save_all_config(self, box=True):
        """Save the current configuration."""
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
                "deepseek_key": self.deepseek_key_var.get(),
                "ollama_url": self.ollama_url_var.get(),
                "ollama_model": self.ollama_model_var.get(),
                "openai_model": self.openai_model_var.get(),
                "openai_url": self.openai_url_var.get(),
                "openai_key": self.openai_key_var.get(),
                "hotwords": self.hotwords,
                "translate_when_sentence_finishes": self.translate_when_sentence_finishes,
            }
        }

        self.config_handler.save_config(current_settings)
        if box:
            messagebox.showinfo("Success", "All settings saved successfully!")

def main():
    root = tk.Tk()
    Real_time_caption_translate = Mainloop(root)
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = Mainloop(root)
    root.mainloop()