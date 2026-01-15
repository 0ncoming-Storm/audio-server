from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx
import torch
from fastapi import Query
import gc
import httpx
import os
from pathlib import Path
import asyncio
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Temporary monkey-patch to disable PyTorch's weights_only safety check
# WARNING: This reduces security — only use if you trust the model sources (Hugging Face pyannote models)
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force disable the safety check
    return _original_load(*args, **kwargs)


torch.load = _patched_load

app = FastAPI(title="Local Audio → Transcript + Summary")

# Global config (adjust to your needs)
WHISPER_MODEL = "large-v3-turbo"  # or large-v2 / medium for less VRAM
BATCH_SIZE = 8  # lower if OOM (4-16 ok on 2070)
COMPUTE_TYPE = "float16"  # "int8" for even lower VRAM
LLM_API_URL = "http://localhost:8080/v1/completions"
LLM_MODEL_NAME = "llama-3.2-3b-instruct-q4_k_m.gguf"

# Load Token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Safety check
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found. Diarization will fail.")


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models once at startup
audio_model = None
diarization_pipeline = None


@app.on_event("startup")
async def startup_event():
    global audio_model, diarization_pipeline
    print("Loading WhisperX model...")
    audio_model = whisperx.load_model(WHISPER_MODEL, device, compute_type=COMPUTE_TYPE)
    print("Loading diarization pipeline...")
    diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    include_summary: bool = Query(
        True, description="Set to false to skip LLM summary generation"
    ),
    summary_mode: str = Query(
        "meeting",
        enum=["meeting", "lecture"],
        description="Choose 'meeting' for action items/minutes or 'lecture' for study notes",
    ),
):
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{file.filename}")
        with temp_path.open("wb") as f:
            f.write(await file.read())

        # 1. Load & preprocess audio
        audio = whisperx.load_audio(str(temp_path))

        # 2. Transcription
        transcribe_result = audio_model.transcribe(audio, batch_size=BATCH_SIZE)
        language = transcribe_result.get("language", "unknown")

        # 3. Word-level alignment
        align_model, metadata = whisperx.load_align_model(
            language_code=transcribe_result["language"], device=device
        )
        result = whisperx.align(
            transcribe_result["segments"], align_model, metadata, audio, device
        )

        # 4. Diarization + assign speakers
        diarize_segments = diarization_pipeline(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format nice transcript
        transcript = ""
        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            start = segment["start"]
            transcript += f"[{start:.1f}s] {speaker}: {text}\n"

        print("Transcript:\n" + transcript)

        # 5. Call llama.cpp server for summary / notes
        summary = ""

        if include_summary:
            async with httpx.AsyncClient() as client:
                # --- MODE SELECTION LOGIC ---
                if summary_mode == "lecture":
                    # LECTURE MODE: Academic / Study Focus
                    system_instruction = (
                        "You are an expert academic tutor. "
                        "Your goal is to create high-quality study notes from a lecture transcript. "
                        "Focus on definitions, dates, formulas, and concepts. "
                        "Ignore filler words and classroom logistics."
                    )
                    user_instruction = (
                        "Analyze the lecture transcript below and generate structured study notes:\n\n"
                        "# [Title]\n\n"
                        "## 1. Core Concepts\n"
                        "- Summarize the main topics.\n\n"
                        "## 2. Key Definitions\n"
                        "- List important terms and definitions.\n\n"
                        "## 3. Potential Exam Questions\n"
                        "- Suggest 3 questions that test understanding of this material.\n\n"
                        f"Transcript:\n{transcript}"
                    )
                    temp = 0.5  # Lower temp for facts

                else:
                    # MEETING MODE: Corporate / Action Focus (Default)
                    system_instruction = (
                        "You are a professional secretary. "
                        "Your goal is to summarize a meeting transcript into concise minutes. "
                        "Focus on decisions made, key discussions, and future tasks."
                    )
                    user_instruction = (
                        "Based on the transcript below, provide:\n"
                        "1. **Meeting Title**\n"
                        "2. **Executive Summary** (3-5 sentences)\n"
                        "3. **Key Bullet Points**\n"
                        "4. **Action Items** (Who is doing what?)\n\n"
                        f"Transcript:\n{transcript}"
                    )
                    temp = 0.6  # Moderate temp for synthesis

                # --- LLAMA 3 PROMPT CONSTRUCTION ---
                final_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_instruction}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_instruction}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

                payload = {
                    "model": LLM_MODEL_NAME,
                    "prompt": final_prompt,
                    "temperature": temp,
                    "max_tokens": 1200,
                    "stop": ["<|eot_id|>"],
                    "stream": False,
                }

                try:
                    response = await client.post(
                        LLM_API_URL, json=payload, timeout=300.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    summary = data.get("choices", [{}])[0].get("text", "").strip()
                except Exception as llm_error:
                    print(f"LLM Error: {llm_error}")
                    summary = f"Error generating summary: {str(llm_error)}"

        return JSONResponse(
            {
                "transcript": transcript,
                "summary": summary,
                "mode": summary_mode,
                "language": language,
                "status": "success",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
