"""TTS playback for transcriptions using Orpheus-FastAPI.

Provides text-to-speech generation and playback via local Orpheus server.
Reuses patterns from ~/.claude/hooks/utils/tts/orpheus_tts.py.
"""

import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_VOICE = "tara"
DEFAULT_TTS_URL = "http://localhost:5005/v1/audio/speech"

# Available Orpheus voices
VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")


def check_server_available(tts_url: str = DEFAULT_TTS_URL) -> bool:
    """Check if Orpheus-FastAPI server is running.

    Args:
        tts_url: TTS API endpoint URL

    Returns:
        True if server is reachable
    """
    # Derive health URL from TTS URL
    base_url = tts_url.rsplit("/v1/", 1)[0]
    health_url = f"{base_url}/health"

    try:
        req = urllib.request.Request(health_url, method="GET")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        # Try base URL as fallback
        try:
            req = urllib.request.Request(base_url, method="GET")
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False


def generate_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    tts_url: str = DEFAULT_TTS_URL,
) -> bytes:
    """Generate speech audio from text using Orpheus-FastAPI.

    Args:
        text: Text to convert to speech
        voice: Orpheus voice name (tara, leah, jess, leo, dan, mia, zac, zoe)
        tts_url: TTS API endpoint URL

    Returns:
        WAV audio bytes

    Raises:
        urllib.error.URLError: If TTS server is unavailable
        urllib.error.HTTPError: If TTS request fails
    """
    payload = json.dumps(
        {
            "model": "orpheus",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": 1.0,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        tts_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def play_audio(audio_bytes: bytes) -> None:
    """Play audio bytes using macOS afplay.

    Args:
        audio_bytes: WAV audio data

    Raises:
        subprocess.CalledProcessError: If playback fails
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        subprocess.run(["afplay", temp_path], check=True, capture_output=True)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def speak(
    text: str,
    voice: str = DEFAULT_VOICE,
    tts_url: str = DEFAULT_TTS_URL,
) -> None:
    """Generate and play speech (convenience wrapper).

    Args:
        text: Text to speak
        voice: Orpheus voice name
        tts_url: TTS API endpoint URL

    Raises:
        Exception: If TTS generation or playback fails
    """
    audio = generate_speech(text, voice, tts_url)
    play_audio(audio)


def cycle_voice(current_voice: str) -> str:
    """Get the next voice in the cycle.

    Args:
        current_voice: Currently selected voice

    Returns:
        Next voice in the VOICES tuple
    """
    try:
        idx = VOICES.index(current_voice)
        return VOICES[(idx + 1) % len(VOICES)]
    except ValueError:
        return VOICES[0]
