"""Terminal UI for real-time transcription.

Simple ANSI-based TUI for voice transcription.
Keyboard controls: SPACE to start/stop, S to save, Q to quit.
"""

import asyncio
import logging
import queue
import select
import sys
import termios
import textwrap
import tty
from datetime import datetime

import numpy as np

from theo.storage.hybrid import HybridStore
from theo.transcription.audio import AudioCapture
from theo.transcription.storage import TranscriptionStorage
from theo.transcription.transcriber import StreamingTranscriber
from theo.transcription.types import TranscriptionSegment, TranscriptionSession

logger = logging.getLogger(__name__)

# ANSI escape codes
CLEAR_SCREEN = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RESET = "\033[0m"


class TranscriptionTUI:
    """Terminal UI for real-time voice transcription.

    Simple ANSI-based display with keyboard controls for
    starting/stopping recording, saving sessions, and quitting.
    """

    def __init__(
        self,
        model_path: str = "mlx-community/whisper-turbo",
        namespace: str = "default",
        language: str | None = None,
        hybrid_store: HybridStore | None = None,
        device: int | None = None,
        initial_prompt: str | None = None,
        batch_mode: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self._namespace = namespace
        self._batch_mode = batch_mode
        self._audio = AudioCapture(device=device)
        self._transcriber = StreamingTranscriber(
            model_path=model_path,
            language=language,
            initial_prompt=initial_prompt,
            batch_mode=batch_mode,
            temperature=temperature,
        )
        self._storage = TranscriptionStorage(hybrid_store) if hybrid_store else None

        self._session: TranscriptionSession | None = None
        self._segments: list[TranscriptionSegment] = []
        self._running = False
        self._recording = False
        self._audio_buffer: list[np.ndarray] = []
        self._message: str = ""

    def run(self) -> None:
        """Run the TUI main loop."""
        self._running = True
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            print("Loading Whisper model...")
            with self._transcriber:
                print("Model loaded!")
                tty.setraw(fd)

                self._draw()
                while self._running:
                    if self._key_available():
                        key = sys.stdin.read(1)
                        if not self._handle_key(key):
                            break
                        self._draw()

                    if self._recording:
                        self._process_audio()
                        self._draw()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print()  # Newline after exit
            if self._recording:
                self._audio.stop()

    def _key_available(self) -> bool:
        """Check if keyboard input is available (non-blocking)."""
        return select.select([sys.stdin], [], [], 0.1)[0] != []

    def _handle_key(self, key: str) -> bool:
        if key == " ":
            self._toggle_recording()
        elif key.lower() == "s":
            self._save_session()
        elif key.lower() == "q":
            return False
        return True

    def _draw(self) -> None:
        """Draw the TUI display using ANSI codes."""
        lines = []
        lines.append(f"{BOLD}═══ Theo Voice Transcription ═══{RESET}")
        lines.append("")

        if self._segments:
            text_parts = [s.text for s in self._segments if s.is_final]
            full_text = " ".join(text_parts)
            wrapped = textwrap.wrap(full_text, width=80) or [""]
            lines.extend(wrapped)
        else:
            lines.append(f"{DIM}Ready to transcribe...{RESET}")

        lines.append("")

        if self._recording:
            mode = "BATCH" if self._batch_mode else "Recording"
            state = f"{mode} (collecting...)" if self._batch_mode else "Recording"
            secs = self._session.duration_seconds() if self._session else 0.0
            duration = self._format_duration(secs)
            mic = self._audio.device_name
        else:
            state = "Stopped"
            duration = "00:00"
            mic = ""

        final_count = sum(1 for s in self._segments if s.is_final)
        status = f"Status: {state} | Duration: {duration} | Segments: {final_count}"
        if mic:
            status += f" | Mic: {mic}"
        lines.append(f"{BOLD}{status}{RESET}")

        if self._message:
            lines.append(f"{GREEN}{self._message}{RESET}")

        lines.append("")
        lines.append(f"{DIM}Controls: [SPACE] Start/Stop | [S] Save | [Q] Quit{RESET}")

        output = CLEAR_SCREEN + "\r\n".join(lines)
        sys.stdout.write(output)
        sys.stdout.flush()

    def _format_duration(self, seconds: float) -> str:
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if not self._recording:
            self._session = TranscriptionSession()
            self._segments = []
            self._audio_buffer = []
            self._message = ""
            try:
                self._audio.start()
                self._recording = True
            except Exception as e:
                self._message = f"Audio error: {e}"
        else:
            self._recording = False
            self._audio.stop()
            if self._session:
                self._session.end_time = datetime.now()
            # Batch mode: transcribe all collected audio after stop
            if self._batch_mode and self._audio_buffer:
                self._message = "Transcribing..."
                self._draw()
                segment = self._transcriber.transcribe_batch(self._audio_buffer)
                self._segments.append(segment)
                if self._session:
                    self._session.add_segment(segment)
                self._audio_buffer = []
                self._message = "Transcription complete"

    def _process_audio(self) -> None:
        """Process audio chunks on main thread (MLX requires main thread)."""
        try:
            chunk = self._audio.get_chunk_nowait()
            if chunk is None:
                return

            # Batch mode: only collect audio, transcribe on stop
            if self._batch_mode:
                self._audio_buffer.append(chunk.data)
                return

            for segment in self._transcriber.transcribe_audio(chunk.data, self._audio_buffer):
                self._segments.append(segment)
                if self._session:
                    self._session.add_segment(segment)

        except queue.Empty:
            pass
        except Exception as e:
            self._message = f"Error: {e}"
            logger.exception("Transcription error: %s", e)

    def _save_session(self) -> None:
        """Save current session to Theo storage."""
        if not self._session:
            self._message = "No session to save"
            return
        if not self._storage:
            self._message = "Storage not configured"
            return
        if self._recording:
            self._toggle_recording()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                memory_id = loop.run_until_complete(
                    self._storage.save_session(self._session, self._namespace)
                )
                self._message = f"Saved: {memory_id[:8]}..."
            finally:
                loop.close()
        except Exception as e:
            self._message = f"Save failed: {e}"
