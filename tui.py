import os
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import requests
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Button,
    Static,
    Label,
    Select,
    Input,
    Switch,
    Markdown,
    TabbedContent,
    TabPane,
    Log,
)
from textual import work
from textual.reactive import reactive


# --- Audio Logic Class ---
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.stream = None
        self.sample_rate = 16000
        self.channels = 1

    def get_devices(self):
        """Returns a list of tuples (label, id) for Textual Select widget."""
        devices = sd.query_devices()
        device_list = []
        try:
            default_in = sd.default.device[0]
        except:
            default_in = -1

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = dev["name"]
                # Mark default device
                label = f"{'★ ' if i == default_in else ''}{name} (ID: {i})"
                device_list.append((label, i))
        return device_list

    def start(self, device_id):
        self.frames = []
        self.recording = True

        # --- Auto-detect supported sample rate ---
        try:
            dev_info = sd.query_devices(device_id, "input")
            self.sample_rate = int(dev_info["default_samplerate"])
        except Exception:
            self.sample_rate = 44100  # Fallback

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=device_id,
            channels=self.channels,
            callback=callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.frames:
            return None

        return np.concatenate(self.frames, axis=0)

    def save_wav(self, audio_data):
        if audio_data is None or audio_data.size == 0:
            return None

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tf.name, self.sample_rate, audio_data.astype(np.float32))
        return tf.name


# --- TUI App ---
class TranscriptionApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #sidebar {
        dock: left;
        width: 35;
        height: 100%;
        background: $panel;
        padding: 1;
        border-right: vkey $accent;
    }

    #main-content {
        height: 100%;
        padding: 1;
    }

    .group {
        margin-bottom: 2;
        background: $boost;
        padding: 1;
        border: tall $background;
    }

    Label {
        color: $text-muted;
        margin-bottom: 1;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        background: $accent;
        color: $text-accent;
        content-align: center middle;
    }

    /* Fixed Height for Buttons */
    Button {
        width: 100%;
        height: 3; 
        margin-bottom: 1;
    }

    /* Default State */
    .record-btn {
        background: $error;
        color: $text;
    }
    
    /* Active/Recording State */
    .record-btn-active {
        background: $error-darken-2;
        color: $text; /* Changed to text color for better readability */
        border: tall $error-lighten-2; 
    }

    Markdown {
        padding: 1;
    }
    
    #url-input {
        margin-bottom: 2;
    }
    """

    TITLE = "WhisperX + Llama Client"
    SUB_TITLE = "Local Transcription & Summarization"

    is_recording = reactive(False)
    status_message = reactive("Ready")

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.temp_file = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Container():
            # LEFT SIDEBAR
            with Vertical(id="sidebar"):
                yield Label("Server URL")
                yield Input(
                    value="http://localhost:8000/process-audio", id="url-input"
                )

                yield Label("Input Device")
                yield Select(
                    options=self.recorder.get_devices(),
                    prompt="Select Mic",
                    id="device-select",
                )

                with Vertical(classes="group"):
                    yield Label("Mode")
                    yield Select(
                        options=[
                            ("Meeting (Minutes)", "meeting"),
                            ("Lecture (Study Notes)", "lecture"),
                        ],
                        value="meeting",
                        allow_blank=False,
                        id="mode-select",
                    )

                    yield Label("Generate Summary?")
                    yield Switch(value=True, id="summary-switch")

                # FIX 1: Added classes="record-btn" so CSS applies immediately
                yield Button(
                    "Start Recording",
                    id="record-btn",
                    variant="error",
                    classes="record-btn",
                )
                yield Button("Quit", id="quit-btn")

            # MAIN CONTENT AREA
            with Vertical(id="main-content"):
                with TabbedContent():
                    with TabPane("Transcript"):
                        yield ScrollableContainer(Markdown("", id="transcript-view"))
                    with TabPane("Summary / Notes"):
                        yield ScrollableContainer(Markdown("", id="summary-view"))
                    with TabPane("Logs"):
                        yield Log(id="log-view", highlight=True)

        yield Static(self.status_message, id="status-bar")
        yield Footer()

    def on_mount(self):
        devices = self.recorder.get_devices()
        if devices:
            self.query_one("#device-select").value = devices[0][1]
        self.log_msg("App initialized.")

    def log_msg(self, text):
        log_view = self.query_one("#log-view", Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_view.write_line(f"[{timestamp}] {text}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.exit()
        elif event.button.id == "record-btn":
            self.toggle_recording()

    def toggle_recording(self):
        btn = self.query_one("#record-btn", Button)

        if not self.is_recording:
            # Start Recording
            device_id = self.query_one("#device-select").value
            if device_id is None:
                self.notify("Please select an input device", severity="error")
                return

            self.is_recording = True
            btn.label = "STOP Recording"

            # FIX 2: Use add_class instead of overwriting classes
            btn.add_class("record-btn-active")

            self.status_message = "Recording... (Press Stop to finish)"
            self.log_msg("Recording started.")
            self.recorder.start(device_id)

        else:
            # Stop Recording
            self.is_recording = False
            btn.label = "Start Recording"

            # FIX 3: Use remove_class to revert to original state
            btn.remove_class("record-btn-active")

            self.status_message = "Processing audio..."
            self.log_msg("Stopping recording...")
            audio_data = self.recorder.stop()
            self.log_msg("Recording stopped.")

            if audio_data is not None:
                self.temp_file = self.recorder.save_wav(audio_data)
                self.upload_and_process()
            else:
                self.log_msg("No audio data captured.")
                self.status_message = "Ready"

    @work(exclusive=True, thread=True)
    def upload_and_process(self):
        url = self.query_one("#url-input").value
        mode = self.query_one("#mode-select").value
        do_summary = self.query_one("#summary-switch").value

        self.app.call_from_thread(self.update_status, "Sending to server...")

        try:
            with open(self.temp_file, "rb") as f:
                files = {"file": (os.path.basename(self.temp_file), f, "audio/wav")}
                params = {
                    "include_summary": str(do_summary).lower(),
                    "summary_mode": mode,
                }

                self.app.call_from_thread(
                    self.log_msg, f"Uploading to {url} ({mode})..."
                )

                response = requests.post(url, files=files, params=params, timeout=300)

            if response.status_code == 200:
                data = response.json()
                self.app.call_from_thread(self.update_results, data)
                self.app.call_from_thread(self.update_status, "Success!")
            else:
                self.app.call_from_thread(
                    self.log_msg, f"Server Error: {response.text}"
                )
                self.app.call_from_thread(self.update_status, "Error")

        except Exception as e:
            self.app.call_from_thread(self.log_msg, f"Connection Error: {e}")
            self.app.call_from_thread(self.update_status, "Connection Failed")

        finally:
            if os.path.exists(self.temp_file):
                os.unlink(self.temp_file)

    def update_status(self, msg):
        self.status_message = msg

    def update_results(self, data):
        raw_transcript = data.get("transcript", "*No transcript returned*")
        formatted_transcript = raw_transcript.replace("\n", "  \n")
        summary = data.get("summary", "*No summary returned*")

        self.query_one("#transcript-view", Markdown).update(formatted_transcript)
        self.query_one("#summary-view", Markdown).update(summary)

        self.log_msg(f"Process complete. Language: {data.get('language')}")
        self.notify("Transcription Complete!")


if __name__ == "__main__":
    app = TranscriptionApp()
    app.run()
