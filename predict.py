"""HiggsAudioM3 — Audio-in, Text-out prediction script.

This script sends an audio file to the HiggsAudioM3 model and prints the
text response. It handles the full pipeline:

    1. Load and chunk the audio (VAD + 4s max chunks)
    2. Build OpenAI-compatible messages with audio_url content parts
    3. Call the API and print the response

Usage:
    # Basic usage — transcribe/process an audio file
    python predict.py path/to/audio.wav

    # Custom system prompt
    python predict.py audio.wav --system-prompt "Summarize this audio."

    # Streaming mode (see tokens as they arrive)
    python predict.py audio.wav --stream

    # Use a different API endpoint
    python predict.py audio.wav --base-url http://localhost:8000/v1

Environment:
    BOSONAI_API_KEY  — API key for authentication (default: "EMPTY")

See README.md for recommended prompts for different use cases (ASR, chat,
tool use, thinking mode).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from audio_utils import chunk_audio_file

# ──────────────────────────────────────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://hackathon.boson.ai/v1"
DEFAULT_MODEL = "higgs-audio-understanding-v3.5-Hackathon"
DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Chat with the user in a friendly and engaging manner. Keep the response concise and natural."

# Stop sequences — these special tokens signal the model to stop generating.
# Required for correct behavior with the HiggsAudioM3 API.
STOP_SEQUENCES = [
    "<|eot_id|>",
    "<|endoftext|>",
    "<|audio_eos|>",
    "<|im_end|>",
]

# Extra parameters for the API backend
EXTRA_BODY = {"skip_special_tokens": False}

# Generation parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048


# ──────────────────────────────────────────────────────────────────────────────
# Build the message payload
# ──────────────────────────────────────────────────────────────────────────────

def build_messages(
    audio_chunks: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_text: str | None = None,
) -> list[dict]:
    """Build OpenAI-format messages with audio chunks embedded.

    The API expects audio as multiple `audio_url` content parts within
    a single user message. Each chunk is base64-encoded WAV data with an
    indexed MIME type: `data:audio/wav_{i};base64,...`

    The index suffix (wav_0, wav_1, ...) tells the API the ordering of
    chunks so it can reconstruct the full audio in the correct sequence.

    Args:
        audio_chunks: List of base64-encoded WAV strings from chunk_audio_file().
        system_prompt: System message to set the model's behavior.
        user_text: Optional text to include alongside the audio (e.g., instructions).

    Returns:
        List of message dicts ready for the OpenAI chat completions API.
    """
    # Build the user message content parts
    user_content: list[dict] = []

    # Add optional text instruction before the audio
    if user_text:
        user_content.append({"type": "text", "text": user_text})

    # Add each audio chunk as an audio_url content part
    # IMPORTANT: The MIME type must be `audio/wav_{i}` (with chunk index),
    # not just `audio/wav`. This is how the API distinguishes multiple chunks.
    for i, chunk_b64 in enumerate(audio_chunks):
        user_content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav_{i};base64,{chunk_b64}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Call the API
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    audio_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_text: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    stream: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send an audio file to HiggsAudioM3 and return the text response.

    This is the main function that ties everything together:
        1. Chunk the audio file using VAD + 4s splitting
        2. Build the message payload
        3. Call the OpenAI-compatible API
        4. Return (or stream) the text response

    Args:
        audio_path:    Path to the audio file (WAV, FLAC, OGG, etc.).
        system_prompt: System message for the model.
        user_text:     Optional text alongside the audio.
        base_url:      API base URL.
        model:         Model name.
        api_key:       API key (defaults to BOSONAI_API_KEY env var or "EMPTY").
        stream:        If True, print tokens as they arrive.
        temperature:   Sampling temperature (0.0 = deterministic, higher = more random).
        top_p:         Nucleus sampling threshold.
        max_tokens:    Maximum tokens to generate.

    Returns:
        The model's text response.
    """
    # ── Step 1: Chunk the audio ──────────────────────────────────────────
    print(f"Loading and chunking audio: {audio_path}")
    audio_chunks, meta = chunk_audio_file(audio_path)
    print(f"  Duration: {meta['duration_s']}s → {meta['num_chunks']} chunks")

    # ── Step 2: Build messages ───────────────────────────────────────────
    messages = build_messages(
        audio_chunks=audio_chunks,
        system_prompt=system_prompt,
        user_text=user_text,
    )

    # ── Step 3: Create the API client ────────────────────────────────────
    resolved_api_key = api_key or os.environ.get("BOSONAI_API_KEY", "EMPTY")
    client = OpenAI(
        base_url=base_url,
        api_key=resolved_api_key,
        timeout=180.0,
        max_retries=3,
    )

    # ── Step 4: Call the API ─────────────────────────────────────────────
    api_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=STOP_SEQUENCES,
        extra_body=EXTRA_BODY,
    )

    if stream:
        # Streaming mode — print tokens as they arrive
        print("\nResponse (streaming):")
        print("-" * 40)
        response_stream = client.chat.completions.create(stream=True, **api_kwargs)
        full_response = ""
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
        print()  # Final newline
        print("-" * 40)
        return full_response.strip()
    else:
        # Non-streaming mode — wait for full response
        response = client.chat.completions.create(**api_kwargs)
        text = ""
        if response.choices and response.choices[0].message:
            text = (response.choices[0].message.content or "").strip()
        print(f"\nResponse:\n{text}")
        return text


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Send audio to HiggsAudioM3 and get a text response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # General chat (audio-in, text-out)
  python predict.py sample.wav

  # ASR (speech recognition) — needs both system and user text prompts
  python predict.py sample.wav \
      --system-prompt "You are an automatic speech recognition (ASR) system." \
      --user-text "Your task is to listen to audio input and output the exact spoken words as plain text in English."

  # Thinking mode (v3.5 only) — adds chain-of-thought reasoning
  python predict.py sample.wav --system-prompt "You are a helpful voice assistant. Use Thinking."

  # Tool use (v3.5 only) — embed tools in system prompt, response contains <tool_call> tags
  python predict.py sample.wav --system-prompt "You are a helpful AI assistant with access to function calling tools. Use the most appropriate tool when needed.\n<tools>{\\\"tools\\\": [{\\\"type\\\": \\\"function\\\", \\\"function\\\": {\\\"name\\\": \\\"get_horoscope\\\", \\\"description\\\": \\\"Get today's horoscope for an astrological sign.\\\", \\\"parameters\\\": {\\\"type\\\": \\\"object\\\", \\\"properties\\\": {\\\"sign\\\": {\\\"type\\\": \\\"string\\\", \\\"description\\\": \\\"An astrological sign like Taurus or Aquarius\\\"}}, \\\"required\\\": [\\\"sign\\\"]}}}]}</tools>"

  # Streaming mode
  python predict.py sample.wav --stream

  # See README.md for full prompt templates and tool use flow
        """,
    )

    parser.add_argument(
        "audio_file",
        help="Path to the audio file (WAV, FLAC, OGG, etc.)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"System prompt for the model (default: '{DEFAULT_SYSTEM_PROMPT}')",
    )
    parser.add_argument(
        "--user-text",
        default=None,
        help="Optional text instruction alongside the audio",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response token by token",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    predict(
        audio_path=args.audio_file,
        system_prompt=args.system_prompt,
        user_text=args.user_text,
        base_url=args.base_url,
        model=args.model,
        stream=args.stream,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
