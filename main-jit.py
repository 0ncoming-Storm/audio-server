from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import whisperx
import torch
import gc
import subprocess
import time
import httpx
import os
import signal
import asyncio
import threading
from pathlib import Path
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# --- MONKEY PATCH ---
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

app = FastAPI(title="Optimized 3B Pipeline")

# --- CONFIGURATION ---
WHISPER_MODEL = "large-v3-turbo"
BATCH_SIZE = 8
COMPUTE_TYPE = "float16"

# Load Token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Safety check
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found. Diarization will fail.")


# PATHS
LLAMA_SERVER_PATH = "llama.cpp/build/bin/llama-server"
LLAMA_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"

# SERVER SETTINGS
LLAMA_PORT = 8081
LLAMA_API_URL = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"

# MEMORY / CONTEXT SETTINGS for 8GB VRAM
# 24576 tokens ~= 90,000 characters (~45 mins of speech).
# This leaves room for the 2GB model + 3.5GB KV Cache + buffer.
LLAMA_CTX_SIZE = "24576"
CHUNK_CHAR_LIMIT = (
    85000  # Set slightly below context limit to leave room for the prompt
)

# TIMEOUTS
BOOT_TIMEOUT = 90  # Increased for allocating large KV cache
GEN_TIMEOUT = 600  # Long generation for full context

device = "cuda" if torch.cuda.is_available() else "cpu"


def cleanup_memory():
    """Aggressively clear VRAM."""
    gc.collect()
    torch.cuda.empty_cache()


async def wait_for_server(base_url, timeout=60):
    start_time = time.time()
    health_url = f"{base_url}/health"
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                resp = await client.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except (httpx.RequestError, httpx.ConnectError):
                pass
            await asyncio.sleep(0.5)
    return False


async def query_llama(messages, temp=0.5):
    """Sends request to llama-server."""
    payload = {
        "messages": messages,
        "temperature": temp,
        "max_tokens": 2000,  # Allow long summaries
        "stream": False,
        "cache_prompt": True,  # Crucial for speed if doing multiple passes
    }

    async with httpx.AsyncClient(timeout=float(GEN_TIMEOUT)) as client:
        response = await client.post(LLAMA_API_URL, json=payload)
        if response.status_code != 200:
            raise Exception(f"LLM API Error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def get_optimized_prompts(mode, transcript_text):
    """
    Returns the System and User messages.
    Uses 'Sandwich Prompting' and XML delimiters for best 3B performance.
    """

    if mode == "lecture":
        # ACADEMIC FRAMEWORK
        system_prompt = (
            "You are an expert academic tutor and note-taker. "
            "Your goal is to convert spoken transcripts into high-quality, structured study materials. "
            "You prioritize accuracy, clear definitions, and exam preparation. "
            "Do not hallucinate information not present in the text."
        )

        user_prompt = f"""
Task: Analyze the following lecture transcript and create a Study Guide.

Instructions:
1. **Title**: Generate a descriptive title.
2. **Executive Summary**: A 3-5 sentence overview of the core thesis.
3. **Core Concepts**: Extract the main topics. Use nested bullet points for details.
4. **Vocabulary & Definitions**: Identify key terms and define them using context from the lecture.

<transcript>
{transcript_text}
</transcript>

REMINDER: Focus on the content inside the <transcript> tags. 
Structure your output using Markdown headers (#, ##, ###).
Start the Study Guide now:
"""

    else:
        # MEETING FRAMEWORK
        system_prompt = (
            "You are a highly efficient Executive Secretary. "
            "Your goal is to distill messy spoken conversation into clean, actionable business minutes. "
            "You focus on decisions, tasks, and deadlines. You ignore small talk and filler."
        )

        user_prompt = f"""
Task: Analyze the following meeting transcript and create Official Minutes.

Instructions:
1. **Meeting Title**: Specific to the topic discussed.
2. **Objective**: What was the primary goal of this discussion?
3. **Key Decisions**: Explicitly state what was agreed upon.
4. **Action Items**: List tasks in this format: "- [ ] [Owner]: [Task Description]".
5. **Open Issues**: What was discussed but left unresolved?

<transcript>
{transcript_text}
</transcript>

REMINDER: Focus on the content inside the <transcript> tags.
Ensure Action Items are clearly assigned if speakers are identified.
Start the Meeting Minutes now:
"""
    return system_prompt, user_prompt


def create_chunks(segments, max_chars=CHUNK_CHAR_LIMIT):
    chunks = []
    current_chunk = ""
    for seg in segments:
        line = f"[{seg['start']:.1f}s] {seg.get('speaker', 'UNK')}: {seg['text'].strip()}\n"
        if len(current_chunk) + len(line) > max_chars:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    include_summary: bool = Query(True),
    summary_mode: str = Query("meeting", enum=["meeting", "lecture"]),
):
    temp_path = Path(f"temp_{file.filename}")
    final_transcript = ""
    final_summary = ""
    server_process = None

    # --- PHASE 1: TRANSCRIPTION ---
    try:
        with temp_path.open("wb") as f:
            f.write(await file.read())

        print("1. [Whisper] Loading...")
        audio = whisperx.load_audio(str(temp_path))
        model = whisperx.load_model(WHISPER_MODEL, device, compute_type=COMPUTE_TYPE)
        print("   [Whisper] Transcribing...")
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]
        del model
        cleanup_memory()

        print("2. [Align] Loading...")
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device
        )
        del align_model, metadata
        cleanup_memory()

        print("3. [Diarize] Loading...")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model
        cleanup_memory()

        # Format Text
        segments = result["segments"]
        for seg in segments:
            final_transcript += f"[{seg['start']:.1f}s] {seg.get('speaker', 'UNK')}: {seg['text'].strip()}\n"

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        cleanup_memory()
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    if include_summary:
        try:
            print("4. [LLM] Booting Llama Server (High Context)...")
            # Updated Arguments for Performance
            server_cmd = [
                LLAMA_SERVER_PATH,
                "-m",
                LLAMA_MODEL_PATH,
                "--port",
                str(LLAMA_PORT),
                "-c",
                LLAMA_CTX_SIZE,  # <--- Using the expanded context
                "-ngl",
                "99",
                "--host",
                "127.0.0.1",
            ]
            # Launch server with stderr captured for live debugging
            server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line-buffered
                universal_newlines=True,  # Text mode for easy line reading
            )

            print(
                f" [LLM] Allocating {LLAMA_CTX_SIZE} tokens KV Cache (this takes a moment)..."
            )
            print(" [LLM] Streaming server logs below for debug:")

            # --- Live streaming of server logs in a background thread ---
            def stream_llama_logs():
                """Reads stderr line-by-line and prints with prefix (non-blocking for main thread)."""
                for line in server_process.stderr:
                    stripped = line.strip()
                    if stripped:  # Skip empty lines
                        print(f"[LLM LOG] {stripped}")

            # Daemon thread so it doesn't block app shutdown
            threading.Thread(target=stream_llama_logs, daemon=True).start()

            # Wait for server to become ready
            is_ready = await wait_for_server(
                f"http://localhost:{LLAMA_PORT}", timeout=BOOT_TIMEOUT
            )
            if not is_ready:
                # On timeout, terminate and collect any remaining logs
                server_process.terminate()
                remaining_logs = server_process.stderr.read()
                if remaining_logs:
                    print("[LLM ERROR LOGS]\n" + remaining_logs)
                raise TimeoutError("Model failed to load within timeout.")

            print(" [LLM] Server Ready. Processing...")

            # --- SINGLE CHUNK PREFERENCE ---
            if len(final_transcript) < CHUNK_CHAR_LIMIT:
                print(" [LLM] Fits in Context. Running Single Pass.")
                sys_msg, user_msg = get_optimized_prompts(
                    summary_mode, final_transcript
                )
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ]
                final_summary = await query_llama(messages, temp=0.5)

            else:
                # --- CHUNKING FALLBACK ---
                print("   [LLM] Transcript exceeds 32k context. Chunking.")
                text_chunks = create_chunks(segments)
                partial_summaries = []

                # Simplified chunk prompt for map-reduce
                for i, chunk in enumerate(text_chunks):
                    print(f"   [LLM] Chunk {i + 1}/{len(text_chunks)}")
                    # We don't need the full fancy prompt for chunks, just raw extraction
                    messages = [
                        {
                            "role": "system",
                            "content": "Summarize the key points of this section.",
                        },
                        {
                            "role": "user",
                            "content": f"Transcript Part {i + 1}:\n{chunk}",
                        },
                    ]
                    partial = await query_llama(messages, temp=0.6)
                    partial_summaries.append(f"--- Part {i + 1} ---\n{partial}")

                print("   [LLM] Merging...")
                combined = "\n\n".join(partial_summaries)
                sys_msg, user_msg = get_optimized_prompts(
                    summary_mode, combined
                )  # Run full framework on the combined notes

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ]
                final_summary = await query_llama(messages, temp=0.5)

        except Exception as e:
            print(f"ERROR: LLM Failure: {e}")
            final_summary = f"Error generating summary: {str(e)}"
            if server_process and server_process.poll() is not None:
                print(f"Server Log: {server_process.stderr.read().decode()}")

        finally:
            if server_process:
                print("   [LLM] Shutting down server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()

    # --- CLEANUP ---
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    cleanup_memory()

    return JSONResponse(
        {
            "transcript": final_transcript,
            "summary": final_summary,
            "status": "success",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
