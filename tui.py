import os
import shutil
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import requests
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Button,
    Label,
    Select,
    Input,
    Switch,
    Markdown,
    TabbedContent,
    TabPane,
    Log,
)
from textual.screen import ModalScreen
from textual import work
from textual.reactive import reactive


# --- Audio Logic Class (Unchanged) ---
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.stream = None
        self.sample_rate = 16000
        self.channels = 1

    def get_devices(self):
        devices = sd.query_devices()
        device_list = []
        try:
            default_in = sd.default.device[0]
        except:
            default_in = -1

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = dev["name"]
                label = f"{'★ ' if i == default_in else ''}{name} (ID: {i})"
                device_list.append((label, i))
        return device_list

    def start(self, device_id):
        self.frames = []
        self.recording = True
        try:
            dev_info = sd.query_devices(device_id, "input")
            self.sample_rate = int(dev_info["default_samplerate"])
        except Exception:
            self.sample_rate = 44100

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


# --- New Save Dialog Screen ---
class SaveScreen(ModalScreen[str]):
    """A modal screen to ask for a filename."""

    CSS = """
    SaveScreen {
        align: center middle;
        background: rgba(0,0,0,0.5);
    }
    
    #save-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        border: heavy $accent;
        background: $surface;
    }

    #save-dialog Label {
        margin-bottom: 1;
        width: 100%;
        text-align: center;
        text-style: bold;
    }

    #save-buttons {
        margin-top: 1;
        align: center middle;
    }

    #save-buttons Button {
        width: 1fr;
        margin: 0 1;
    }
    """

    def __init__(self, default_filename: str):
        super().__init__()
        self.default_filename = default_filename

    def compose(self) -> ComposeResult:
        with Vertical(id="save-dialog"):
            yield Label("Save Transcript As:")
            yield Input(value=self.default_filename, id="filename-input")
            with Horizontal(id="save-buttons"):
                yield Button("Save", variant="success", id="confirm-save")
                yield Button("Cancel", variant="error", id="cancel-save")

    def on_mount(self):
        self.query_one("#filename-input").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-save":
            filename = self.query_one("#filename-input").value
            self.dismiss(filename)
        elif event.button.id == "cancel-save":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


# --- TUI App ---
class TranscriptionApp(App):
    CSS = """
    /* --- Layout & Colors --- */
    Screen {
        layout: horizontal;
        background: $background;
    }

    /* --- Sidebar --- */
    #sidebar-container {
        width: 40;
        height: 100%;
        background: $panel;
        border-right: vkey $accent;
        dock: left;
    }
    
    .section {
        background: $surface;
        border: tall $background;
        margin: 1 1 0 1;
        padding: 1;
        height: auto;
    }

    .section-audio-box {
        background: $surface;
        border: tall $background;
        margin: 1 1 0 1;
        padding: 1;
        height: 50%;
    }

    .section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    /* --- Controls --- */
    Label {
        color: $text-muted;
        margin-top: 1;
    }

    Input {
        border: tall $background;
        background: $boost;
    }
    
    Select {
        border: tall $background;
    }

    /* Compact Switch Row */
    .switch-row {
        height: 3;
        align: left middle;
        margin-top: 1;
    }
    .switch-row Label {
        width: 1fr;
        margin-top: 0;
        padding-top: 1;
    }
    
    /* --- Buttons --- */
    Button {
        width: 100%;
        margin-top: 1;
    }
    
    .record-btn {
        background: $error;
        color: $text;
        text-style: bold;
    }
    
    .record-btn-active {
        background: $error-darken-2;
        border: heavy $error-lighten-2;
        color: $text;
        text-style: bold;
    }

    .hidden {
        display: none;
    }

    /* --- Main Content --- */
    #main-content {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    TabbedContent {
        height: 1fr;
    }

    /* Markdown Styling */
    Markdown {
        padding: 1 2;
        background: $surface;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $accent;
        color: $text-accent;
        padding: 0 1;
    }
    """

    TITLE = "WhisperX Client"
    SUB_TITLE = "Transcription & Summarization"

    is_recording = reactive(False)
    status_message = reactive("Ready")
    transcript_tmp_path = None

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.audio_temp_file = None

    def compose(self) -> ComposeResult:
        yield Header()

        # --- LEFT SIDEBAR (ScrollableContainer allows it to fit on small screens) ---
        with ScrollableContainer(id="sidebar-container"):
            # 1. Connection Section
            with Vertical(classes="section"):
                yield Label("SERVER CONFIG", classes="section-title")
                yield Label("Endpoint URL:")
                yield Input(value="http://localhost:8000/process-audio", id="url-input")

            # 2. Input Logic Section
            with Vertical(classes="section-audio-box"):
                yield Label("AUDIO SOURCE", classes="section-title")
                yield Select(
                    options=[("Microphone", "mic"), ("Local File Upload", "file")],
                    value="mic",
                    allow_blank=False,
                    id="source-select",
                )

                # -- Sub-group: Microphone Controls --
                with Vertical(id="mic-controls"):
                    yield Label("Microphone Device:")
                    yield Select(
                        options=self.recorder.get_devices(),
                        prompt="Default Device",
                        id="device-select",
                    )
                    with Horizontal(classes="switch-row"):
                        yield Label("Save Audio Backup?")
                        yield Switch(value=False, id="save-local-switch")

                    yield Button(
                        "START RECORDING", id="record-btn", classes="record-btn"
                    )

                # -- Sub-group: File Controls --
                with Vertical(id="file-controls", classes="hidden"):
                    yield Label("File Path:")
                    yield Input(placeholder="/path/to/audio.wav", id="file-path-input")
                    yield Button("UPLOAD FILE", id="upload-file-btn", variant="primary")

            # 3. Processing Options Section
            with Vertical(classes="section"):
                yield Label("PROCESSING OPTIONS", classes="section-title")
                yield Label("Transcription Mode:")
                yield Select(
                    options=[
                        ("Meeting (Minutes)", "meeting"),
                        ("Lecture (Notes)", "lecture"),
                    ],
                    value="meeting",
                    allow_blank=False,
                    id="mode-select",
                )
                with Horizontal(classes="switch-row"):
                    yield Label("Generate Summary?")
                    yield Switch(value=True, id="summary-switch")

            # 4. Actions Section
            with Vertical(classes="section"):
                yield Label("ACTIONS", classes="section-title")
                yield Button(
                    "Save Transcript to File", id="save-transcript-btn", disabled=True
                )
                yield Button("Quit Application", id="quit-btn", variant="default")

        # --- RIGHT MAIN CONTENT ---
        with Vertical(id="main-content"):
            with TabbedContent():
                with TabPane("Transcript"):
                    yield ScrollableContainer(
                        Markdown("*Waiting for input...*", id="transcript-view")
                    )
                with TabPane("Summary"):
                    yield ScrollableContainer(
                        Markdown("*Waiting for summary...*", id="summary-view")
                    )
                with TabPane("System Logs"):
                    yield Log(id="log-view", highlight=True)

            yield Label(self.status_message, id="status-bar")

        yield Footer()

    # --- Event Handlers & Logic ---

    def on_mount(self):
        devices = self.recorder.get_devices()
        if devices:
            self.query_one("#device-select").value = devices[0][1]
        self.log_msg("App initialized. Select source and begin.")

    def log_msg(self, text):
        log_view = self.query_one("#log-view", Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_view.write_line(f"[{timestamp}] {text}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.control.id == "source-select":
            mic_controls = self.query_one("#mic-controls")
            file_controls = self.query_one("#file-controls")

            if event.value == "mic":
                mic_controls.remove_class("hidden")
                file_controls.add_class("hidden")
            else:
                mic_controls.add_class("hidden")
                file_controls.remove_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.exit()
        elif event.button.id == "record-btn":
            self.toggle_recording()
        elif event.button.id == "upload-file-btn":
            self.handle_local_file_upload()
        elif event.button.id == "save-transcript-btn":
            self.save_transcript_to_disk()

    def handle_local_file_upload(self):
        path_input = self.query_one("#file-path-input", Input)
        file_path = path_input.value.strip()

        # Simple string cleanup (remove quotes if user dragged/dropped)
        file_path = file_path.replace('"', "").replace("'", "")

        if not file_path:
            self.notify("Please enter a file path.", severity="error")
            return

        if not os.path.exists(file_path):
            self.notify(f"File not found: {file_path}", severity="error")
            return

        self.audio_temp_file = file_path
        self.log_msg(f"Selected local file: {file_path}")
        self.upload_and_process(is_temp=False)

    def toggle_recording(self):
        btn = self.query_one("#record-btn", Button)
        if not self.is_recording:
            # Start
            device_id = self.query_one("#device-select").value
            if device_id is None:
                self.notify("Please select an input device", severity="error")
                return
            self.is_recording = True
            btn.label = "STOP RECORDING"
            btn.add_class("record-btn-active")
            self.status_message = "Recording in progress..."
            self.log_msg("Microphone recording started.")
            self.recorder.start(device_id)
        else:
            # Stop
            self.is_recording = False
            btn.label = "START RECORDING"
            btn.remove_class("record-btn-active")
            self.status_message = "Processing audio..."
            self.log_msg("Stopping recording...")
            audio_data = self.recorder.stop()
            self.log_msg("Recording stopped.")

            if audio_data is not None:
                self.audio_temp_file = self.recorder.save_wav(audio_data)
                self.log_msg(
                    f"Captured audio saved to temp: {os.path.basename(self.audio_temp_file)}"
                )

                # Optional Local Backup
                save_local = self.query_one("#save-local-switch").value
                if save_local:
                    recordings_dir = "recordings"
                    os.makedirs(recordings_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    permanent_path = os.path.join(
                        recordings_dir, f"recording_{timestamp}.wav"
                    )
                    shutil.copy(self.audio_temp_file, permanent_path)
                    self.log_msg(f"Backup saved to: {permanent_path}")

                self.upload_and_process(is_temp=True)
            else:
                self.log_msg("No audio data captured.")
                self.status_message = "Ready"

    @work(exclusive=True, thread=True)
    def upload_and_process(self, is_temp=True):
        url = self.query_one("#url-input").value
        mode = self.query_one("#mode-select").value
        do_summary = self.query_one("#summary-switch").value

        self.app.call_from_thread(self.update_status, "Uploading & Transcribing...")

        try:
            if not self.audio_temp_file:
                return

            with open(self.audio_temp_file, "rb") as f:
                filename = os.path.basename(self.audio_temp_file)
                files = {"file": (filename, f, "audio/wav")}

                params = {
                    "include_summary": str(do_summary).lower(),
                    "summary_mode": mode,
                }

                self.app.call_from_thread(
                    self.log_msg, f"Sending {filename} to {url}..."
                )
                response = requests.post(url, files=files, params=params, timeout=300)

            if response.status_code == 200:
                data = response.json()
                self.app.call_from_thread(self.update_results, data)
                self.app.call_from_thread(
                    self.update_status, "Success! Transcription Ready."
                )
            else:
                self.app.call_from_thread(
                    self.log_msg,
                    f"Server Error {response.status_code}: {response.text}",
                )
                self.app.call_from_thread(
                    self.update_status, "Server returned an error."
                )

        except Exception as e:
            self.app.call_from_thread(self.log_msg, f"Connection Failure: {e}")
            self.app.call_from_thread(self.update_status, "Connection Failed.")

        finally:
            if (
                is_temp
                and self.audio_temp_file
                and os.path.exists(self.audio_temp_file)
            ):
                os.unlink(self.audio_temp_file)
                self.audio_temp_file = None

    def update_status(self, msg):
        self.status_message = msg
        self.query_one("#status-bar", Label).update(msg)

    def update_results(self, data):
        raw_transcript = data.get("transcript", "*No transcript returned*")

        # 1. Save full transcript to temp file
        tf = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        )
        tf.write(raw_transcript)
        tf.close()
        self.transcript_tmp_path = tf.name
        self.log_msg(f"Full transcript buffer: {self.transcript_tmp_path}")

        # 2. Preview Logic (First 50 lines)
        lines = raw_transcript.splitlines()
        preview_text = "\n".join(lines[:50])
        if len(lines) > 50:
            preview_text += f"\n\n... [Truncated. Total lines: {len(lines)}. Save file to view all.] ..."

        formatted_preview = f"```text\n{preview_text}\n```"

        # 3. Update UI
        self.query_one("#transcript-view", Markdown).update(formatted_preview)

        summary = data.get("summary", "*No summary returned*")
        self.query_one("#summary-view", Markdown).update(summary)

        self.query_one("#save-transcript-btn").disabled = False
        self.notify("Processing Complete")

    def save_transcript_to_disk(self):
        """Opens the SaveScreen modal to let the user choose a location."""
        if not self.transcript_tmp_path or not os.path.exists(self.transcript_tmp_path):
            self.notify("No transcript to save.", severity="error")
            return

        # Create a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"transcript_{timestamp}.txt"

        def handle_save(save_path: str):
            """Callback for when the SaveScreen closes."""
            if not save_path:
                return  # User cancelled

            # Handle relative paths or just filenames
            save_path = os.path.abspath(save_path)

            try:
                # Ensure directory exists if they typed a long path
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                shutil.copy(self.transcript_tmp_path, save_path)
                self.log_msg(f"Saved: {save_path}")
                self.notify(f"File saved to: {os.path.basename(save_path)}")
            except Exception as e:
                self.log_msg(f"Save Error: {e}")
                self.notify(f"Error saving file: {e}", severity="error")

        self.push_screen(SaveScreen(default_name), handle_save)


if __name__ == "__main__":
    app = TranscriptionApp()
    app.run()
