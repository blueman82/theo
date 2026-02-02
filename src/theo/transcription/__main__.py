"""Entry point for Theo voice transcription.

Run with: python -m theo.transcription

Modes:
- TUI mode (default): Interactive terminal UI with keyboard controls
- Oneshot mode: When stdout is piped, records until silence, prints transcription
  Example: python -m theo.transcription --batch | pbcopy
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from theo.config import TheoSettings
from theo.storage.hybrid import HybridStore
from theo.transcription import audio_storage, tts
from theo.transcription.audio import AudioCapture
from theo.transcription.storage import TranscriptionStorage
from theo.transcription.transcriber import StreamingTranscriber
from theo.transcription.tui import TranscriptionTUI

# Load settings - whisper_model is required from .env
# Note: TheoSettings loads from env vars, pyright doesn't understand pydantic_settings
_settings = TheoSettings()  # type: ignore[call-arg]


def list_devices() -> None:
    """Print available input devices."""
    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = ">" if i == sd.default.device[0] else " "
            print(f"  {marker} {i}: {dev['name']}")
    print("\nUse --device N to select a device")


def list_transcriptions(storage: TranscriptionStorage, namespace: str | None) -> None:
    """List saved transcriptions."""
    records = storage.list_transcriptions(namespace=namespace, limit=50)
    if not records:
        print("No transcriptions found.")
        return

    print(f"{'ID':<12} {'Date':<20} {'Duration':>10} {'Audio':>6} {'Text':<40}")
    print("-" * 92)

    for rec in records:
        # Format date
        dt = datetime.fromtimestamp(rec.created_at)
        date_str = dt.strftime("%Y-%m-%d %H:%M")

        # Format duration
        if rec.duration_seconds:
            mins = int(rec.duration_seconds) // 60
            secs = int(rec.duration_seconds) % 60
            dur_str = f"{mins:02d}:{secs:02d}"
        else:
            dur_str = "--:--"

        # Check audio availability
        has_audio = "Yes" if rec.audio_path else "No"

        # Truncate text
        text = rec.full_text[:37] + "..." if len(rec.full_text) > 40 else rec.full_text
        text = text.replace("\n", " ")

        print(f"{rec.id[:12]:<12} {date_str:<20} {dur_str:>10} {has_audio:>6} {text:<40}")


def play_transcription(
    storage: TranscriptionStorage,
    session_id: str,
    voice: str,
    tts_url: str,
) -> None:
    """Play transcription via TTS."""
    # Try to find transcription by prefix
    records = storage.list_transcriptions(limit=100)
    matching = [r for r in records if r.id.startswith(session_id)]

    if not matching:
        print(f"Transcription not found: {session_id}")
        return

    if len(matching) > 1:
        print(f"Multiple matches for '{session_id}':")
        for rec in matching:
            print(f"  {rec.id}")
        return

    rec = matching[0]
    print(f"Playing transcription {rec.id[:8]}... via TTS (voice: {voice})")
    print(f"Text: {rec.full_text[:100]}{'...' if len(rec.full_text) > 100 else ''}")

    try:
        tts.speak(rec.full_text, voice=voice, tts_url=tts_url)
        print("Playback complete.")
    except Exception as e:
        print(f"TTS error: {e}")


def replay_transcription(storage: TranscriptionStorage, session_id: str) -> None:
    """Replay original audio for a transcription."""
    # Try to find transcription by prefix
    records = storage.list_transcriptions(limit=100)
    matching = [r for r in records if r.id.startswith(session_id)]

    if not matching:
        print(f"Transcription not found: {session_id}")
        return

    if len(matching) > 1:
        print(f"Multiple matches for '{session_id}':")
        for rec in matching:
            print(f"  {rec.id}")
        return

    rec = matching[0]
    if not rec.audio_path:
        print(f"No audio file for transcription {rec.id[:8]}...")
        print("Use --play instead for TTS playback.")
        return

    audio_path = Path(rec.audio_path)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Replaying original audio for {rec.id[:8]}...")
    try:
        audio_storage.play_audio_file(audio_path)
        print("Replay complete.")
    except Exception as e:
        print(f"Replay error: {e}")


def delete_transcription(storage: TranscriptionStorage, session_id: str) -> None:
    """Delete a transcription and its audio file."""
    # Try to find transcription by prefix
    records = storage.list_transcriptions(limit=100)
    matching = [r for r in records if r.id.startswith(session_id)]

    if not matching:
        print(f"Transcription not found: {session_id}")
        return

    if len(matching) > 1:
        print(f"Multiple matches for '{session_id}':")
        for rec in matching:
            print(f"  {rec.id}")
        return

    rec = matching[0]
    print(f"Deleting transcription {rec.id}...")

    if storage.delete_transcription(rec.id):
        print("Deleted successfully.")
    else:
        print("Delete failed.")


def run_oneshot(
    model_path: str,
    device: int | None,
    language: str | None,
    prompt: str,
    temperature: float,
    silence_timeout: float = 2.0,
    max_duration: float = 120.0,
) -> None:
    """Record until silence, transcribe, print to stdout.

    Used when stdout is piped (not a TTY). Status messages go to stderr.
    """
    print("Loading model...", file=sys.stderr)

    transcriber = StreamingTranscriber(
        model_path=model_path,
        language=language,
        initial_prompt=prompt,
        batch_mode=True,
        temperature=temperature,
    )

    audio = AudioCapture(device=device)
    audio_buffer: list[np.ndarray] = []

    with transcriber:
        print("Recording... (silence stops)", file=sys.stderr)
        audio.start()

        speech_started = False
        last_audio_time = time.time()
        start_time = time.time()

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    print("\nMax duration reached", file=sys.stderr)
                    break

                try:
                    chunk = audio.get_chunk_nowait()
                    if chunk is not None:
                        audio_buffer.append(chunk.data)
                        speech_started = True
                        last_audio_time = time.time()
                except Exception:
                    pass

                if speech_started:
                    silence_duration = time.time() - last_audio_time
                    if silence_duration >= silence_timeout:
                        break

                time.sleep(0.05)

        finally:
            audio.stop()

        if not audio_buffer:
            print("No audio captured", file=sys.stderr)
            return

        print("Transcribing...", file=sys.stderr)
        segment = transcriber.transcribe_batch(audio_buffer)
        print(segment.text)


def main() -> None:
    """Run the transcription TUI or execute commands."""
    parser = argparse.ArgumentParser(description="Theo Voice Transcription")
    parser.add_argument(
        "--model",
        default=_settings.whisper_model,
        help="HuggingFace model path for Whisper (from THEO_WHISPER_MODEL)",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Storage namespace for saved sessions",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for transcription (e.g., 'en')",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (use --list-devices to see available)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Clear dictation with proper punctuation.",
        help="Initial prompt to guide transcription style (reduces hallucinations)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode for maximum accuracy (transcribes after stop, not real-time)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for batch mode (0.0 = deterministic)",
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Disable saving to Theo storage (transcription only)",
    )
    parser.add_argument(
        "--voice",
        default="tara",
        choices=tts.VOICES,
        help="TTS voice for playback (default: tara)",
    )
    parser.add_argument(
        "--tts-url",
        default="http://localhost:5005/v1/audio/speech",
        help="Orpheus-FastAPI TTS endpoint URL",
    )

    # Command flags
    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved transcriptions and exit",
    )
    parser.add_argument(
        "--play",
        metavar="SESSION_ID",
        help="Play transcription via TTS (can use ID prefix)",
    )
    parser.add_argument(
        "--replay",
        metavar="SESSION_ID",
        help="Replay original audio recording (can use ID prefix)",
    )
    parser.add_argument(
        "--delete",
        metavar="SESSION_ID",
        help="Delete transcription and audio file (can use ID prefix)",
    )

    args = parser.parse_args()

    # Handle list-devices first (no storage needed)
    if args.list_devices:
        list_devices()
        return

    # Create storage for commands that need it
    hybrid_store: HybridStore | None = None
    storage: TranscriptionStorage | None = None

    if not args.no_storage or args.list or args.play or args.replay or args.delete:
        hybrid_store = asyncio.run(HybridStore.create())
        storage = TranscriptionStorage(hybrid_store)

    try:
        # Handle commands
        if args.list:
            if storage:
                list_transcriptions(
                    storage, args.namespace if args.namespace != "default" else None
                )
            return

        if args.play:
            if storage:
                play_transcription(storage, args.play, args.voice, args.tts_url)
            return

        if args.replay:
            if storage:
                replay_transcription(storage, args.replay)
            return

        if args.delete:
            if storage:
                delete_transcription(storage, args.delete)
            return

        # Oneshot mode when stdout is piped (not a TTY)
        if not sys.stdout.isatty():
            run_oneshot(
                model_path=args.model,
                device=args.device,
                language=args.language,
                prompt=args.prompt,
                temperature=args.temperature,
            )
            return

        # Run TUI mode
        tui = TranscriptionTUI(
            model_path=args.model,
            namespace=args.namespace,
            language=args.language,
            hybrid_store=hybrid_store,
            device=args.device,
            initial_prompt=args.prompt,
            batch_mode=args.batch,
            temperature=args.temperature,
            tts_voice=args.voice,
            tts_url=args.tts_url,
        )
        tui.run()

    finally:
        if hybrid_store is not None:
            hybrid_store.close()


if __name__ == "__main__":
    main()
