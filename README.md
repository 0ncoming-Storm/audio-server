# Audio Server

**Local Audio Transcription & Summarization Pipeline**

A high-performance audio processing server that combines WhisperX for accurate transcription with speaker diarization and Llama 3.2 for intelligent summarization. Perfect for converting meeting recordings and lectures into structured, actionable documents.

## ✨ Features

- 🎤 **Accurate Transcription**: Powered by WhisperX (large-v3-turbo model)
- 👥 **Speaker Diarization**: Automatic speaker identification and labeling
- 🧠 **AI Summarization**: Local LLM processing with Llama 3.2 3B
- 📝 **Dual Modes**:
  - **Meeting Mode**: Generates action items, decisions, and minutes
  - **Lecture Mode**: Creates study notes, definitions, and exam questions
- 🖥️ **TUI Client**: Beautiful terminal interface for recording and processing
- 🚀 **FastAPI Backend**: RESTful API for integration with other tools
- 🔒 **100% Local**: All processing happens on your machine - no cloud dependencies

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: ~10GB for models

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **CUDA**: 12.x or compatible
- **Python**: 3.10 or 3.11
- **uv**: Python package manager (installation instructions below)

## 🚀 Installation

### 1. Install uv Package Manager

`uv` is a fast Python package manager that will handle all dependencies:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```bash
git clone https://github.com/0ncoming-Storm/audio-server.git
cd audio-server
```

### 3. Install Python Dependencies

Using `uv`, install all required packages:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from requirements.txt
uv pip install -r requirements.txt
```

**Note**: The installation includes PyTorch with CUDA support. This may take several minutes depending on your internet connection.

### 4. Set Up Hugging Face Token

Speaker diarization requires a Hugging Face token with access to pyannote models:

1. Create a Hugging Face account at https://huggingface.co
2. Accept the user agreements for these models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Generate an access token at https://huggingface.co/settings/tokens
4. Create a `.secrets` file or set environment variable:

```bash
# Option 1: Create .secrets file (already in .gitignore)
echo "HF_TOKEN=hf_your_token_here" > .secrets

# Option 2: Export environment variable
export HF_TOKEN=hf_your_token_here
```

### 5. Install llama.cpp (for Summarization)

The JIT version (`main-jit.py`) requires llama.cpp server:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support
make LLAMA_CUDA=1

# Download Llama 3.2 3B model (Q4_K_M quantization recommended)
mkdir -p ../models
cd ../models

# Download from Hugging Face
# Example: Using huggingface-cli
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  llama-3.2-3b-instruct-q4_k_m.gguf \
  --local-dir . \
  --local-dir-use-symlinks False
```

**Note**: Adjust model paths in `main-jit.py` if you use different locations.

## 🎯 Usage

### Option 1: Basic Server (External LLM)

This version expects an already-running llama.cpp server:

```bash
# Start the server
python main.py

# The API will be available at http://localhost:8000
```

### Option 2: JIT Server (Built-in LLM Management)

This version automatically starts/stops the llama.cpp server for each request:

```bash
# Start the server
python main-jit.py

# Update paths in the file if needed:
# LLAMA_SERVER_PATH = "/path/to/llama.cpp/llama-server"
# LLAMA_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
```

### Option 3: TUI Client (Recommended for Interactive Use)

The Terminal User Interface provides an easy way to record and process audio:

```bash
python tui.py
```

**TUI Features**:
- Select your microphone from a list of available devices
- Record audio with a single button press
- Choose between Meeting and Lecture modes
- View real-time transcription and summaries
- Logs panel for debugging

## 📡 API Documentation

### POST `/process-audio`

Process an audio file and generate transcript + summary.

**Parameters**:
- `file` (form-data, required): Audio file (WAV, MP3, M4A, etc.)
- `include_summary` (query, optional): Generate summary (default: `true`)
- `summary_mode` (query, optional): Mode selection - `meeting` or `lecture` (default: `meeting`)

**Example Request**:

```bash
curl -X POST "http://localhost:8000/process-audio?include_summary=true&summary_mode=meeting" \
  -F "file=@recording.wav"
```

**Example Response**:

```json
{
  "transcript": "[0.0s] SPEAKER_00: Hello everyone...\n[5.2s] SPEAKER_01: Thanks for joining...",
  "summary": "# Meeting Minutes\n\n## Executive Summary\n...",
  "language": "en",
  "mode": "meeting",
  "status": "success"
}
```

## ⚙️ Configuration

### Server Configuration (main.py)

Edit the global configuration section:

```python
WHISPER_MODEL = "large-v3-turbo"  # Options: large-v3-turbo, large-v2, medium
BATCH_SIZE = 8                     # Lower if OOM (4-16 typical)
COMPUTE_TYPE = "float16"           # "int8" for less VRAM
LLM_API_URL = "http://localhost:8080/v1/completions"  # Your llama.cpp endpoint
LLM_MODEL_NAME = "llama-3.2-3b-instruct-q4_k_m.gguf"
```

### JIT Server Configuration (main-jit.py)

```python
LLAMA_SERVER_PATH = "/home/storm/llama.cpp/llama-server"  # Path to binary
LLAMA_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
LLAMA_CTX_SIZE = "24576"  # Context window size (tokens)
CHUNK_CHAR_LIMIT = 85000  # Character limit per chunk
```

## 🐛 Troubleshooting

### CUDA Out of Memory Errors

If you encounter OOM errors:

1. **Reduce batch size**: Set `BATCH_SIZE = 4` in the config
2. **Use smaller model**: Change to `WHISPER_MODEL = "medium"`
3. **Use INT8**: Set `COMPUTE_TYPE = "int8"`
4. **Reduce context**: In JIT mode, lower `LLAMA_CTX_SIZE = "16384"`

### HF_TOKEN Not Found Warning

```
WARNING: HF_TOKEN not found in .secrets file. Diarization will fail.
```

**Solution**: Create a `.secrets` file with your Hugging Face token:
```bash
echo "HF_TOKEN=hf_your_token_here" > .secrets
```

Or export the environment variable before running:
```bash
export HF_TOKEN=hf_your_token_here
python main.py
```

### Diarization Model Download Issues

If pyannote models fail to download:

1. Ensure you've accepted the model licenses on Hugging Face
2. Verify your HF_TOKEN has read access
3. Check your internet connection
4. Try manually downloading with: `huggingface-cli login`

### TUI Audio Device Not Found

If the TUI can't find your microphone:

1. Install sounddevice dependencies:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libportaudio2
   ```
2. List available devices:
   ```python
   import sounddevice as sd
   print(sd.query_devices())
   ```
3. Check device permissions (Linux):
   ```bash
   sudo usermod -a -G audio $USER
   # Log out and back in
   ```

### llama.cpp Server Won't Start

Check the following:

1. **Path is correct**: Verify `LLAMA_SERVER_PATH` points to the built binary
2. **Model exists**: Confirm `LLAMA_MODEL_PATH` is valid
3. **Port is free**: Ensure port 8081 (or configured port) isn't in use
4. **CUDA libraries**: Make sure CUDA toolkit is properly installed

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   TUI       │─────▶│  FastAPI Server  │─────▶│  WhisperX   │
│  Client     │      │   (main.py)      │      │  (GPU)      │
│  (tui.py)   │◀─────│                  │◀─────│             │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              │
                              ▼
                     ┌──────────────────┐
                     │  llama.cpp       │
                     │  Server (LLM)    │
                     │  (GPU)           │
                     └──────────────────┘
```

### Processing Pipeline

1. **Audio Input**: Upload via API or record via TUI
2. **Transcription**: WhisperX converts speech to text
3. **Alignment**: Word-level timestamps generated
4. **Diarization**: Speakers identified and labeled
5. **Summarization**: LLM generates structured output
6. **Response**: JSON with transcript and summary

## 📦 Included Files

- `main.py` - FastAPI server (external LLM mode)
- `main-jit.py` - FastAPI server (auto-managed LLM mode)
- `tui.py` - Terminal UI client with recording
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## 🔐 Security Notes

⚠️ **Important**: The code includes a PyTorch monkey-patch (`weights_only=False`) to load pyannote models. This is only safe because the models come from the trusted Hugging Face pyannote organization. Do not use this approach with untrusted model sources.

## 📄 License

This project uses several open-source components:
- WhisperX (BSD License)
- llama.cpp (MIT License)
- PyAnnote Audio (MIT License)
- FastAPI (MIT License)

Please review individual component licenses for details.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for additional languages
- WebSocket streaming for real-time transcription
- Docker containerization
- Web UI frontend
- Batch processing mode

## 🙏 Acknowledgments

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [PyAnnote](https://github.com/pyannote/pyannote-audio) team
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the Troubleshooting section above
- Review llama.cpp and WhisperX documentation

---

**Made with ❤️ for local, privacy-focused AI**
