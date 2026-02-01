#!/usr/bin/env python3
"""Test script for voice recording and playback quality evaluation.

Records your voice, then plays it back so you can hear the quality.
This helps decide: keep original audio playback, or just use Orpheus TTS?

Usage:
    uv run python scripts/test_voice_playback.py

Controls:
    Press ENTER to start recording
    Press ENTER again to stop and play back
    Press Q to quit
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000  # Standard for speech
CHANNELS = 1


def record_audio() -> np.ndarray:
    """Record audio until user presses Enter."""
    print("\n Recording... Press ENTER to stop.")

    frames: list[np.ndarray] = []
    recording = True

    def callback(indata: np.ndarray, _frame_count: int, _time_info: Any, _status: sd.CallbackFlags) -> None:
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        callback=callback,
    )

    with stream:
        input()  # Wait for Enter
        recording = False

    if not frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(frames, axis=0)


def play_audio(audio: np.ndarray) -> None:
    """Play back recorded audio using sounddevice."""
    print("Playing back...")
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Playback complete.")


def play_audio_afplay(audio: np.ndarray) -> None:
    """Play back using macOS afplay (same method as Orpheus TTS)."""
    print("Playing back via afplay...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        temp_path = f.name

    try:
        subprocess.run(["afplay", temp_path], check=True)
        print("Playback complete.")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main() -> None:
    print("=" * 50)
    print("Voice Recording & Playback Quality Test")
    print("=" * 50)
    print("\nThis test helps you evaluate audio quality.")
    print("Record your voice, then hear it played back.")
    print("\nIf quality is good → keep original audio playback option")
    print("If quality is bad → just use Orpheus TTS with voices")
    print()

    # Show available devices
    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = ">" if i == sd.default.device[0] else " "
            print(f"  {marker} {i}: {dev['name']}")
    print()

    while True:
        print("-" * 50)
        cmd = input("Press ENTER to record, or Q to quit: ").strip().lower()

        if cmd == "q":
            print("Goodbye!")
            break

        # Record
        audio = record_audio()

        if len(audio) == 0:
            print("No audio recorded.")
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"Recorded {duration:.1f} seconds")

        # Play back with both methods
        print("\n--- Method 1: sounddevice (direct) ---")
        play_audio(audio)

        time.sleep(0.5)

        print("\n--- Method 2: afplay (same as Orpheus TTS) ---")
        play_audio_afplay(audio)

        print("\nHow did it sound? Good quality = keep audio playback option")
        print()


if __name__ == "__main__":
    main()
