"""Audio file storage for transcription recordings.

Saves and plays original audio recordings to ~/.theo/audio/.
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_DIR = Path.home() / ".theo" / "audio"


def save_audio(
    session_id: str,
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    audio_dir: Path | None = None,
) -> Path:
    """Save audio buffer to WAV file.

    Args:
        session_id: Unique session identifier for filename
        audio_data: Audio samples as numpy array (mono)
        sample_rate: Audio sample rate (default: 16000 Hz)
        audio_dir: Override audio directory (default: ~/.theo/audio)

    Returns:
        Path to saved WAV file
    """
    target_dir = audio_dir or AUDIO_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{session_id}.wav"
    sf.write(path, audio_data, sample_rate)
    return path


def play_audio_file(audio_path: Path) -> None:
    """Play WAV file via afplay (macOS).

    Args:
        audio_path: Path to WAV file

    Raises:
        FileNotFoundError: If audio file doesn't exist
        subprocess.CalledProcessError: If playback fails
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    subprocess.run(["afplay", str(audio_path)], check=True, capture_output=True)


def delete_audio(audio_path: Path) -> bool:
    """Delete audio file if it exists.

    Args:
        audio_path: Path to audio file

    Returns:
        True if file was deleted, False if it didn't exist
    """
    if audio_path.exists():
        audio_path.unlink()
        return True
    return False


def get_audio_path(session_id: str, audio_dir: Path | None = None) -> Path:
    """Get the expected path for a session's audio file.

    Args:
        session_id: Session identifier
        audio_dir: Override audio directory

    Returns:
        Path where the audio file would be stored
    """
    target_dir = audio_dir or AUDIO_DIR
    return target_dir / f"{session_id}.wav"
