"""Eigen AI TTS wrapper — voice-cloned speech generation using Higgs Audio V2.5."""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

EIGEN_API_URL = "https://api-web.eigenai.com/api/v1/generate"
EIGEN_MODEL = "higgs2p5"
DEFAULT_SAMPLING = json.dumps({"temperature": 1.0, "top_p": 0.95, "top_k": 50})


def generate_speech(
    text: str,
    voice_reference_path: str,
    output_path: str,
    api_key: str | None = None,
) -> str:
    """Generate speech with voice cloning using Eigen AI TTS.

    Args:
        text: Text to speak.
        voice_reference_path: Path to WAV file for voice cloning.
        output_path: Path to save generated WAV file.
        api_key: Eigen AI API key. Defaults to EIGEN_AI_API_KEY env var.

    Returns:
        Path to the generated audio file.

    Raises:
        requests.HTTPError: If the TTS API call fails.
    """
    resolved_key = api_key or os.environ.get("EIGEN_AI_API_KEY", "")
    headers = {"Authorization": f"Bearer {resolved_key}"}

    with open(voice_reference_path, "rb") as ref_audio:
        files = {"voice_reference_file": ("reference.wav", ref_audio, "audio/wav")}
        data = {
            "model": EIGEN_MODEL,
            "text": text,
            "stream": "false",
            "sampling": DEFAULT_SAMPLING,
        }
        response = requests.post(
            EIGEN_API_URL, headers=headers, data=data, files=files, timeout=120
        )

    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
