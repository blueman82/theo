"""Terminal UI for real-time transcription.

Simple ANSI-based TUI for voice transcription.
Keyboard controls: SPACE to start/stop, S to save, P play TTS, R replay, V voice, Q quit.
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
from pathlib import Path

import numpy as np

from theo.storage.hybrid import HybridStore
from theo.transcription import audio_storage, tts
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
YELLOW = "\033[33m"
RESET = "\033[0m"


class TranscriptionTUI:
    """Terminal UI for real-time voice transcription.

    Simple ANSI-based display with keyboard controls for
    starting/stopping recording, saving sessions, playback, and quitting.
    """

    def __init__(
        self,
        model_path: str,
        namespace: str = "default",
        language: str | None = None,
        hybrid_store: HybridStore | None = None,
        device: int | None = None,
        initial_prompt: str | None = None,
        batch_mode: bool = False,
        temperature: float = 0.0,
        tts_voice: str = "tara",
        tts_url: str = "http://localhost:5005/v1/audio/speech",
    ) -> None:
        self._namespace = namespace
        self._batch_mode = batch_mode
        self._model_path = model_path
        self._language = language
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

        # TTS and playback state
        self._tts_voice = tts_voice
        self._tts_url = tts_url
        self._last_saved_id: str | None = None
        self._last_audio_path: Path | None = None

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
        elif key.lower() == "m":
            self._memorize_session()
        elif key.lower() == "p":
            self._play_tts()
        elif key.lower() == "r":
            self._replay_original()
        elif key.lower() == "v":
            self._cycle_voice()
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

        # Voice indicator
        lines.append(f"{YELLOW}Voice: {self._tts_voice}{RESET}")

        if self._message:
            lines.append(f"{GREEN}{self._message}{RESET}")

        lines.append("")
        controls = (
            "[SPACE] Start/Stop | [S] Save | [M] Memorize | "
            "[P] Play TTS | [R] Replay | [V] Voice | [Q] Quit"
        )
        lines.append(f"{DIM}Controls: {controls}{RESET}")

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
            self._last_saved_id = None
            self._last_audio_path = None
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
                # Keep audio buffer for save, don't clear here
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
        """Save current session as transcription only (source, no memory)."""
        if not self._session:
            self._message = "No session to save"
            return
        if not self._storage:
            self._message = "Storage not configured"
            return
        if self._last_saved_id == self._session.id:
            self._message = "Already saved"
            return
        if self._recording:
            self._toggle_recording()

        try:
            # Combine audio buffer into single array
            audio_data: np.ndarray | None = None
            if self._audio_buffer:
                audio_data = np.concatenate(self._audio_buffer)

            # Save to SQLite with audio file (transcription only, no memory)
            transcription_id = self._storage.save_session_to_db(
                session=self._session,
                audio_data=audio_data,
                namespace=self._namespace,
                model_used=self._model_path,
                language=self._language,
            )
            self._last_saved_id = transcription_id

            # Track audio path for replay
            if audio_data is not None:
                self._last_audio_path = audio_storage.get_audio_path(self._session.id)
            else:
                self._last_audio_path = None

            self._message = f"Saved: {transcription_id[:8]}... (source only)"

            # Clear audio buffer after save
            self._audio_buffer = []

        except Exception as e:
            self._message = f"Save failed: {e}"
            logger.exception("Save error: %s", e)

    def _memorize_session(self) -> None:
        """Save current session as transcription AND memory (source + knowledge)."""
        if not self._session:
            self._message = "No session to memorize"
            return
        if not self._storage:
            self._message = "Storage not configured"
            return
        if self._recording:
            self._toggle_recording()

        # First save the transcription if not already saved
        already_saved = self._last_saved_id == self._session.id
        if not already_saved:
            try:
                audio_data: np.ndarray | None = None
                if self._audio_buffer:
                    audio_data = np.concatenate(self._audio_buffer)

                transcription_id = self._storage.save_session_to_db(
                    session=self._session,
                    audio_data=audio_data,
                    namespace=self._namespace,
                    model_used=self._model_path,
                    language=self._language,
                )
                self._last_saved_id = transcription_id

                if audio_data is not None:
                    self._last_audio_path = audio_storage.get_audio_path(self._session.id)
                else:
                    self._last_audio_path = None

                self._audio_buffer = []
            except Exception as e:
                self._message = f"Save failed: {e}"
                logger.exception("Save error: %s", e)
                return

        # Now create memory with provenance link
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Add source_transcription to metadata for provenance
                self._session.metadata["source_transcription_id"] = self._session.id
                memory_id = loop.run_until_complete(
                    self._storage.save_session(self._session, self._namespace)
                )
                self._message = f"Memorized: {self._session.id[:8]}... → memory: {memory_id[:8]}..."
            finally:
                loop.close()
        except Exception as e:
            self._message = f"Memorize failed: {e}"
            logger.exception("Memorize error: %s", e)

    def _play_tts(self) -> None:
        """Play current transcription text via TTS."""
        if not self._segments:
            self._message = "No transcription to play"
            return
        if self._recording:
            self._message = "Stop recording first"
            return

        text_parts = [s.text for s in self._segments if s.is_final]
        full_text = " ".join(text_parts)
        if not full_text.strip():
            self._message = "No text to play"
            return

        self._message = f"Playing TTS ({self._tts_voice})..."
        self._draw()

        try:
            tts.speak(full_text, voice=self._tts_voice, tts_url=self._tts_url)
            self._message = "TTS playback complete"
        except Exception as e:
            self._message = f"TTS error: {e}"
            logger.exception("TTS error: %s", e)

    def _replay_original(self) -> None:
        """Replay original recorded audio."""
        if self._recording:
            self._message = "Stop recording first"
            return

        # Check if we have a recently saved audio path
        if self._last_audio_path and self._last_audio_path.exists():
            self._message = "Replaying original audio..."
            self._draw()
            try:
                audio_storage.play_audio_file(self._last_audio_path)
                self._message = "Replay complete"
            except Exception as e:
                self._message = f"Replay error: {e}"
                logger.exception("Replay error: %s", e)
            return

        # Otherwise check if we have unsaved audio buffer
        if self._audio_buffer:
            self._message = "Save first to replay original audio"
            return

        self._message = "No audio to replay (save a session first)"

    def _cycle_voice(self) -> None:
        """Cycle through available TTS voices."""
        self._tts_voice = tts.cycle_voice(self._tts_voice)
        self._message = f"Voice: {self._tts_voice}"
