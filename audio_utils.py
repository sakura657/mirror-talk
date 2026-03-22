"""Audio utilities for HiggsAudioM3 — VAD chunking pipeline.

Audio must be split into chunks of at most 4 seconds before sending to the API.
This is a requirement of the HiggsAudioM3 API.

Pipeline overview:
    1. Load audio file → waveform + sample rate
    2. Resample to 16kHz (required by the API)
    3. Run Silero VAD to detect speech segments
    4. Fill gaps between VAD segments (so the full audio is covered)
    5. Enforce max 4s per chunk (split longer segments)
    6. Encode each chunk as base64 WAV

Usage:
    from audio_utils import chunk_audio_file
    chunks = chunk_audio_file("my_audio.wav")
    # chunks is a list of base64-encoded WAV strings, each ≤ 4 seconds
"""

import base64
import io
import wave
from typing import Any

import numpy as np
import soundfile as sf

# ──────────────────────────────────────────────────────────────────────────────
# Constants — required by the HiggsAudioM3 API
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE = 16_000          # The API expects 16kHz audio
MAX_CHUNK_SECONDS = 4.0              # Maximum audio length per chunk accepted by the API
MAX_CHUNK_SAMPLES = int(MAX_CHUNK_SECONDS * TARGET_SAMPLE_RATE)  # 64,000 samples
MIN_CHUNK_SAMPLES = 1_600            # ~0.1s — server rejects audio shorter than this

# Silero VAD parameters
VAD_THRESHOLD = 0.55                 # Speech probability threshold
VAD_MIN_SPEECH_MS = 125              # Minimum speech duration to keep
VAD_MIN_SILENCE_MS = 200             # Minimum silence duration to split on
VAD_SPEECH_PAD_MS = 300              # Padding added around speech segments


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load audio from file
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return (waveform, sample_rate).

    Supports any format that libsndfile handles (WAV, FLAC, OGG, etc.).
    Stereo audio is mixed down to mono by averaging channels.

    NOTE: Any audio loading library works here — soundfile, librosa,
    torchaudio, scipy.io.wavfile, etc. — as long as the output is:
      - waveform: 1-D numpy float32 array with values in [-1, 1]
      - sample_rate: integer sample rate
    For example:
      - librosa:    data, sr = librosa.load(file_path, sr=None, mono=True)
      - torchaudio: wv, sr = torchaudio.load(file_path); data = wv.mean(0).numpy()

    Args:
        file_path: Path to the audio file.

    Returns:
        waveform: 1-D numpy float32 array with values in [-1, 1].
        sample_rate: Original sample rate of the file.
    """
    data, sr = sf.read(file_path, dtype="float32")

    # Mix stereo → mono by averaging channels
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, sr


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Resample to 16kHz
# ──────────────────────────────────────────────────────────────────────────────

def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample audio to the target sample rate.

    The API expects 16kHz audio. If your audio is already 16kHz,
    this function returns the input unchanged.

    NOTE: Either resampling method works fine — the only requirement is
    that the final audio is 16kHz. Two options are shown below:
      - torchaudio.transforms.Resample (higher quality sinc interpolation)
      - np.interp (lightweight, no torchaudio dependency)

    Args:
        waveform:  1-D numpy float32 array.
        orig_sr:   Original sample rate.
        target_sr: Desired sample rate (default: 16,000).

    Returns:
        Resampled 1-D numpy float32 array.
    """
    if orig_sr == target_sr:
        return waveform

    # Option A: torchaudio (higher quality sinc interpolation)
    import torch
    import torchaudio
    wv_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    resampled = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)(wv_tensor)
    return resampled.squeeze(0).numpy()

    # Option B: numpy linear interpolation (no torchaudio needed)
    # Uncomment the lines below and comment out Option A if you prefer fewer dependencies.
    # n_out = int(len(waveform) * target_sr / orig_sr)
    # indices = np.linspace(0, len(waveform) - 1, n_out)
    # return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Silero VAD — detect speech segments
# ──────────────────────────────────────────────────────────────────────────────

_vad_model: Any = None  # Cached singleton


def _get_silero_vad() -> Any:
    """Load Silero VAD model (cached after first call)."""
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    from silero_vad import load_silero_vad
    _vad_model = load_silero_vad(onnx=True, opset_version=16)
    return _vad_model


def get_vad_chunks(waveform: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Run Silero VAD and return speech segment boundaries as (start, end) sample indices.

    If no speech is detected, returns a single segment covering the entire audio.
    This ensures we never drop audio — even silence gets sent to the API.

    Args:
        waveform: 1-D numpy float32 array (must be 16kHz).
        sr:       Sample rate (should be 16000).

    Returns:
        List of (start_sample, end_sample) tuples for each speech segment.
    """
    import torch
    from silero_vad import get_speech_timestamps

    model = _get_silero_vad()
    wv_tensor = torch.tensor(waveform, dtype=torch.float32)

    timestamps = get_speech_timestamps(
        audio=wv_tensor,
        model=model,
        threshold=VAD_THRESHOLD,
        sampling_rate=sr,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        visualize_probs=False,
        return_seconds=False,
    )

    # Fallback: if VAD finds no speech, treat the entire audio as one segment
    if not timestamps:
        return [(0, len(waveform))]

    return [(ts["start"], ts["end"]) for ts in timestamps]


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Fill gaps between VAD segments
# ──────────────────────────────────────────────────────────────────────────────

def fill_vad_gaps(
    vad_chunks: list[tuple[int, int]],
    total_samples: int,
) -> list[tuple[int, int]]:
    """Expand VAD segments to cover the full audio with no gaps.

    Why: VAD only returns speech regions, but the API should receive the
    complete audio (including brief silences between speech). This step
    extends each chunk to start where the previous one ended, and extends
    the last chunk to the end of the audio.

    Args:
        vad_chunks:    List of (start, end) from VAD.
        total_samples: Total number of samples in the audio.

    Returns:
        List of (start, end) covering the entire audio without overlaps.
    """
    filled: list[tuple[int, int]] = []
    prev_end = 0

    for idx, (start, end) in enumerate(vad_chunks):
        # Extend start backwards to cover the gap from previous chunk
        chunk_start = min(prev_end, start)
        # Last chunk extends to end of audio
        chunk_end = total_samples if idx == len(vad_chunks) - 1 else end
        filled.append((chunk_start, chunk_end))
        prev_end = end

    return filled


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Enforce max 4-second chunk length
# ──────────────────────────────────────────────────────────────────────────────

def enforce_max_chunk_len(
    chunks: list[tuple[int, int]],
    max_samples: int = MAX_CHUNK_SAMPLES,
) -> list[tuple[int, int]]:
    """Split any chunk exceeding the maximum length into sub-chunks.

    Why: The API accepts at most 4 seconds of audio per chunk.
    Chunks longer than 4s (64,000 samples at 16kHz) must be split into
    sequential sub-chunks of at most max_samples.

    Args:
        chunks:      List of (start, end) sample index tuples.
        max_samples: Maximum samples per chunk (default: 64,000 = 4s at 16kHz).

    Returns:
        List of (start, end) where every chunk has (end - start) ≤ max_samples.
    """
    result: list[tuple[int, int]] = []

    for start, end in chunks:
        length = end - start
        if length == 0:
            continue

        # If chunk fits within limit, keep as-is
        if length <= max_samples:
            result.append((start, end))
        else:
            # Split into sub-chunks of max_samples
            pos = start
            while pos < end:
                next_pos = min(end, pos + max_samples)
                result.append((pos, next_pos))
                pos = next_pos

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Encode a waveform chunk to base64 WAV
# ──────────────────────────────────────────────────────────────────────────────

def encode_chunk_to_base64(waveform_chunk: np.ndarray, sr: int) -> str:
    """Encode a waveform chunk as a base64-encoded WAV string (16-bit PCM).

    Uses Python's stdlib `wave` module — no soundfile needed for encoding.

    Args:
        waveform_chunk: 1-D numpy float32 array (values in [-1, 1]).
        sr:             Sample rate.

    Returns:
        Base64-encoded WAV string.
    """
    # Convert float32 [-1, 1] → int16 [-32768, 32767]
    samples_i16 = np.clip(waveform_chunk * 32767, -32768, 32767).astype(np.int16)
    raw_bytes = samples_i16.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sr)
        wf.writeframes(raw_bytes)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# High-level entry point: file → list of base64 chunks
# ──────────────────────────────────────────────────────────────────────────────

def chunk_audio_file(file_path: str) -> tuple[list[str], dict]:
    """Load an audio file and split it into VAD-segmented, 4s-max base64 chunks.

    This is the main function you'll use. It runs the full pipeline:
        load → resample → VAD → fill gaps → enforce 4s max → encode

    Args:
        file_path: Path to an audio file (WAV, FLAC, OGG, etc.).

    Returns:
        chunks: List of base64-encoded WAV strings, each ≤ 4 seconds.
        metadata: Dict with debug info (duration, sample rate, chunk boundaries).

    Example:
        chunks, meta = chunk_audio_file("interview.wav")
        print(f"Split into {len(chunks)} chunks from {meta['duration_s']}s audio")
    """
    # 1. Load audio
    waveform, sr = load_audio(file_path)

    # 2. Resample to 16kHz
    waveform = resample_audio(waveform, sr, TARGET_SAMPLE_RATE)
    sr = TARGET_SAMPLE_RATE

    total_samples = len(waveform)

    # 3. Run VAD to find speech segments
    vad_raw = get_vad_chunks(waveform, sr)

    # 4. Fill gaps so the entire audio is covered
    vad_filled = fill_vad_gaps(vad_raw, total_samples)

    # 5. Split any chunk longer than 4 seconds
    vad_final = enforce_max_chunk_len(vad_filled)

    # 6. Encode each chunk as base64 WAV, padding short chunks to minimum length
    encoded_chunks: list[str] = []
    for start, end in vad_final:
        chunk = waveform[start:end]

        # Pad very short chunks with silence (server rejects < 0.1s audio)
        if len(chunk) < MIN_CHUNK_SAMPLES:
            pad = np.zeros(MIN_CHUNK_SAMPLES - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        encoded_chunks.append(encode_chunk_to_base64(chunk, sr))

    # Build metadata for debugging
    metadata = {
        "duration_s": round(total_samples / sr, 3),
        "sample_rate": sr,
        "num_chunks": len(vad_final),
        "vad_raw_segments": len(vad_raw),
        "chunk_boundaries": vad_final,
    }

    return encoded_chunks, metadata
