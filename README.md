# MirrorTalk

Hear yourself, but kinder.

MirrorTalk transforms angry speech into calm, compassionate speech in the speaker's own cloned voice. It uses **HiggsAudio M3 v3.5** for emotion analysis + NVC (Non-Violent Communication) rewriting, and **Higgs Audio V2.5** (served via Eigen AI) for voice cloning TTS.

## How It Works

1. Record or upload an audio clip of angry/frustrated speech
2. M3 v3.5 analyzes the audio — transcribes, identifies the emotion and unmet need
3. The message is rewritten using Non-Violent Communication principles
4. Higgs Audio V2.5 speaks the rewritten text back in your own voice, but calmer

## Setup

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), ffmpeg

```bash
# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure API keys — create a .env file:
BOSONAI_API_KEY=your-boson-ai-key
EIGEN_AI_API_KEY=your-eigen-ai-key
```

## Run

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 in your browser.

## Tech Stack

- **Backend:** FastAPI + Python
- **Audio Understanding:** HiggsAudio M3 v3.5 (BosonAI)
- **Voice Cloning TTS:** Higgs Audio V2.5 (Eigen AI)
- **Frontend:** Vanilla HTML/CSS/JS
- **Audio Processing:** pydub, Silero VAD, torchaudio
