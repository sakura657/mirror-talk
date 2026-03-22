"""MirrorTalk — FastAPI backend.

Orchestrates M3 v3.5 (emotion analysis + NVC rewrite) and Eigen AI TTS (voice cloning)
to transform angry speech into calm, compassionate speech in the speaker's own voice.
"""

import os
import re
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

# NOTE: We re-implement the M3 API call here instead of calling predict() directly
# because predict() prints to stdout (pollutes server logs) and returns only a string.
# We reuse build_messages and constants from predict.py but handle the API call ourselves.
from audio_utils import chunk_audio_file
from predict import build_messages, DEFAULT_MODEL, EXTRA_BODY, STOP_SEQUENCES
from tts_utils import generate_speech

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

NVC_SYSTEM_PROMPT = """You are an emotion decoupling expert. Listen to the user's audio carefully.

Respond with EXACTLY these 4 labeled sections, one per line. Do NOT add any other text or explanation.

TRANSCRIPTION: [Write the exact words the speaker said]
EMOTION: [Identify the underlying emotion in one short sentence]
NEED: [What they actually need/feel beneath the anger in one short sentence]
REWRITE: [Rewrite the message using Non-Violent Communication — state observation without judgment, express the feeling beneath the anger, identify the unmet need, and make a clear gentle request. Keep the same meaning and intent but transform the tone to be calm and compassionate. Output in the same language as the speaker.]"""

MAX_DURATION_S = 30

app = FastAPI(title="MirrorTalk")


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any pydub-supported audio format to 16-bit PCM WAV."""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")


def _check_duration(wav_path: str) -> float:
    """Return duration in seconds. Raises HTTPException if too long."""
    audio = AudioSegment.from_wav(wav_path)
    duration_s = len(audio) / 1000.0
    if duration_s > MAX_DURATION_S:
        raise HTTPException(status_code=400, detail=f"Audio must be under {MAX_DURATION_S} seconds (got {duration_s:.1f}s)")
    return duration_s


def _call_m3(wav_path: str) -> str:
    """Call M3 v3.5 with Thinking mode and return the raw response text."""
    audio_chunks, meta = chunk_audio_file(wav_path)
    messages = build_messages(audio_chunks=audio_chunks, system_prompt=NVC_SYSTEM_PROMPT)

    api_key = os.environ.get("BOSONAI_API_KEY", "EMPTY")
    client = OpenAI(
        base_url="https://hackathon.boson.ai/v1",
        api_key=api_key,
        timeout=180.0,
        max_retries=3,
    )

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        stop=STOP_SEQUENCES,
        extra_body=EXTRA_BODY,
    )

    if response.choices and response.choices[0].message:
        return (response.choices[0].message.content or "").strip()
    return ""


def _parse_response(response: str) -> dict:
    """Parse M3 structured response into 4 components."""
    # Extract content from <think> block if present, then use output after it.
    # If ALL content is inside <think> (nothing after), use the think content itself.
    think_match = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        after_think = think_match.group(2).strip()
        text = after_think if after_think else think_content
    else:
        text = response.strip()
    # Strip wrapping quotes if the entire response is quoted
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    def extract(label: str, next_labels: list[str]) -> str:
        next_part = "|".join(r"\*?\*?" + nl + r":?\*?\*?" for nl in next_labels)
        if next_part:
            pattern = r"\*?\*?" + label + r":?\*?\*?\s*(.+?)(?:" + next_part + "|$)"
        else:
            pattern = r"\*?\*?" + label + r":?\*?\*?\s*(.+)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip().strip('"') if m else ""

    transcription = extract("TRANSCRIPTION", ["EMOTION"])
    emotion = extract("EMOTION", ["NEED"])
    need = extract("NEED", ["REWRITE", "NON-VIOLENT", "NVC"])
    rewrite = extract("REWRITE", [])
    if not rewrite:
        rewrite = extract("NON-VIOLENT COMMUNICATION REWRITTEN MESSAGE", [])
    if not rewrite:
        rewrite = extract("NVC", [])
    if not rewrite:
        # Fallback: everything after the last labeled section
        rewrite = text

    return {
        "transcription": transcription,
        "emotion": emotion,
        "need": need,
        "rewrite": rewrite,
    }


@app.post("/api/transform")
async def transform(audio: UploadFile = File(...)):
    # Save uploaded file to temp
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        # Convert to WAV if needed
        if not suffix.lower().endswith(".wav"):
            wav_path = tmp_path + ".wav"
            try:
                _convert_to_wav(tmp_path, wav_path)
            except Exception:
                raise HTTPException(status_code=400, detail="Unsupported audio format")

        # Check duration
        _check_duration(wav_path)

        # Step 1: M3 emotion analysis + NVC rewrite
        try:
            raw_response = _call_m3(wav_path)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Speech analysis failed: {e}")

        parsed = _parse_response(raw_response)

        # Step 2: TTS voice cloning
        output_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        try:
            generate_speech(
                text=parsed["rewrite"],
                voice_reference_path=wav_path,
                output_path=output_path,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Voice generation failed: {e}")

        return JSONResponse({
            "transcription": parsed["transcription"],
            "emotion": parsed["emotion"],
            "need": parsed["need"],
            "rewrite": parsed["rewrite"],
            "audio_url": f"/audio/{output_filename}",
        })
    finally:
        # Clean up temp files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    if not re.match(r"^[a-zA-Z0-9_\-]+\.wav$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
