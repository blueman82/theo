"""Entry point for Theo voice transcription TUI.

Run with: python -m theo.transcription
"""

import argparse
import asyncio

import sounddevice as sd

from theo.storage.hybrid import HybridStore
from theo.transcription.tui import TranscriptionTUI


def list_devices() -> None:
    """Print available input devices."""
    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = ">" if i == sd.default.device[0] else " "
            print(f"  {marker} {i}: {dev['name']}")
    print("\nUse --device N to select a device")


def main() -> None:
    """Run the transcription TUI."""
    parser = argparse.ArgumentParser(description="Theo Voice Transcription")
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-turbo",
        help="HuggingFace model path for Whisper",
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
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    hybrid_store: HybridStore | None = None
    if not args.no_storage:
        hybrid_store = asyncio.run(HybridStore.create())

    tui = TranscriptionTUI(
        model_path=args.model,
        namespace=args.namespace,
        language=args.language,
        hybrid_store=hybrid_store,
        device=args.device,
        initial_prompt=args.prompt,
        batch_mode=args.batch,
        temperature=args.temperature,
    )
    try:
        tui.run()
    finally:
        if hybrid_store is not None:
            hybrid_store.close()


if __name__ == "__main__":
    main()
