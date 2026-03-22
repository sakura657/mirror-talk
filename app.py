"""MirrorTalk — FastAPI backend.

Orchestrates M3 v3.5 (emotion analysis + NVC rewrite) and Eigen AI TTS (voice cloning)
to transform angry speech into calm, compassionate speech in the speaker's own voice.
"""

import base64
import os
import re
import tempfile
import uuid

import requests as http_requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
VIDEO_DIR = os.path.join(OUTPUTS_DIR, "video")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

EIGEN_BASE_URL = "https://api-web.eigenai.com"

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


# ── Video Generation Pipeline ────────────────────────


class VideoRequest(BaseModel):
    rewrite: str
    emotion: str


def _build_image_prompt(rewrite: str, emotion: str) -> str:
    """Build a descriptive image prompt from the rewrite and emotion."""
    return (
        f"A peaceful, artistic scene that visually represents the emotion of {emotion}. "
        f"The scene conveys the message: '{rewrite[:120]}'. "
        "Soft natural lighting, warm colors, cinematic composition, "
        "high quality, no text, no words, no letters."
    )


def _build_video_prompt(rewrite: str, emotion: str) -> str:
    """Build a video animation prompt from the rewrite and emotion."""
    return (
        f"Gentle, slow cinematic motion. The scene softly comes alive, "
        f"conveying a feeling of {emotion}. Subtle movement like swaying, "
        f"floating particles, or gentle light changes. Peaceful and calming."
    )


@app.post("/api/generate-video")
async def generate_video(req: VideoRequest):
    """Step 1: Generate image from text, then submit video generation job."""
    eigen_key = os.environ.get("EIGEN_AI_API_KEY", "")
    headers = {"Authorization": f"Bearer {eigen_key}"}

    # Step 1: Generate image
    image_prompt = _build_image_prompt(req.rewrite, req.emotion)
    try:
        img_resp = http_requests.post(
            f"{EIGEN_BASE_URL}/api/v1/generate",
            headers={**headers, "Content-Type": "application/json"},
            json={"model": "eigen-image", "prompt": image_prompt},
            timeout=120,
        )
        img_resp.raise_for_status()
        img_data = img_resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {e}")

    b64 = img_data.get("turbo_image_base64")
    if not b64:
        raise HTTPException(status_code=502, detail="No image returned from API")

    # Save image
    image_id = uuid.uuid4().hex[:8]
    image_filename = f"img_{image_id}.png"
    image_path = os.path.join(VIDEO_DIR, image_filename)
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(b64))

    # Step 2: Submit video generation job
    video_prompt = _build_video_prompt(req.rewrite, req.emotion)
    try:
        with open(image_path, "rb") as image_file:
            vid_resp = http_requests.post(
                f"{EIGEN_BASE_URL}/api/v1/generate",
                headers={"Authorization": f"Bearer {eigen_key}"},
                data={
                    "model": "wan2p2-i2v-14b-turbo",
                    "prompt": video_prompt,
                    "infer_steps": "5",
                    "seed": "42",
                },
                files={"image": (image_filename, image_file, "image/png")},
                timeout=120,
            )
            vid_resp.raise_for_status()
            vid_data = vid_resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Video job submission failed: {e}")

    task_id = vid_data.get("task_id")
    if not task_id:
        raise HTTPException(status_code=502, detail="No task_id returned from video API")

    return JSONResponse({
        "task_id": task_id,
        "image_url": f"/video/{image_filename}",
    })


@app.get("/api/video-status")
async def video_status(taskId: str):
    """Poll video generation status and download result when complete."""
    eigen_key = os.environ.get("EIGEN_AI_API_KEY", "")
    headers = {"Authorization": f"Bearer {eigen_key}"}

    try:
        status_resp = http_requests.get(
            f"{EIGEN_BASE_URL}/api/v1/generate/status",
            params={"jobId": taskId, "model": "wan2p2-i2v-14b-turbo"},
            headers=headers,
            timeout=30,
        )
        status_resp.raise_for_status()
        status_data = status_resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Status check failed: {e}")

    status = status_data.get("status")

    if status == "completed":
        # Download the video
        try:
            video_resp = http_requests.get(
                f"{EIGEN_BASE_URL}/api/v1/generate/result",
                params={"jobId": taskId, "model": "wan2p2-i2v-14b-turbo"},
                headers=headers,
                timeout=120,
            )
            video_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Video download failed: {e}")

        video_filename = f"vid_{uuid.uuid4().hex[:8]}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        with open(video_path, "wb") as f:
            f.write(video_resp.content)

        return JSONResponse({
            "status": "completed",
            "video_url": f"/video/{video_filename}",
        })
    elif status == "failed":
        return JSONResponse({
            "status": "failed",
            "error": status_data.get("error", "Unknown error"),
        })
    else:
        return JSONResponse({"status": "processing"})


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve generated images and videos."""
    if not re.match(r"^[a-zA-Z0-9_\-]+\.(mp4|png)$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
    return FileResponse(path, media_type=media_type)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
