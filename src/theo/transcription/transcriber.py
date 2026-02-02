"""Streaming transcription using mlx-audio Whisper.

Provides real-time transcription from audio capture using mlx-audio's
streaming Whisper model with lazy loading for efficient resource usage.
"""

import warnings
from collections.abc import Generator
from types import TracebackType
from typing import Any

import mlx.core as mx
import numpy as np

from theo.transcription.audio import AudioCapture
from theo.transcription.types import TranscriptionSegment


class StreamingTranscriber:
    """Streaming transcription using mlx-audio Whisper.

    Lazy-loads the Whisper model on first use and provides real-time
    transcription from AudioCapture streams.

    Example:
        with StreamingTranscriber() as transcriber:
            with AudioCapture() as capture:
                capture.start()
                for segment in transcriber.transcribe_stream(capture):
                    print(segment.text)
    """

    def __init__(
        self,
        model_path: str = "mlx-community/whisper-large-v3-mlx",
        chunk_duration: float = 1.0,
        frame_threshold: int = 25,
        language: str | None = None,
        initial_prompt: str | None = None,
        batch_mode: bool = False,
        temperature: float = 0.0,
    ) -> None:
        """Initialize streaming transcriber.

        Args:
            model_path: HuggingFace model path for Whisper
            chunk_duration: Duration for streaming chunks in seconds
            frame_threshold: Frame threshold for streaming detection
            language: Language code for transcription, None for auto-detect
            initial_prompt: Guide transcription style, reduces hallucinations
            batch_mode: If True, use batch processing for max accuracy (no real-time)
            temperature: Sampling temperature for batch mode (0.0 = deterministic)
        """
        self._model_path = model_path
        self._model: Any | None = None
        self._chunk_duration = chunk_duration
        self._frame_threshold = frame_threshold
        self._language = language
        self._initial_prompt = initial_prompt
        self._batch_mode = batch_mode
        self._temperature = temperature

    def _ensure_model(self) -> Any:
        """Lazy-load the Whisper model.

        Returns:
            Loaded Whisper model instance
        """
        if self._model is None:
            warnings.filterwarnings("ignore", message="Could not load WhisperProcessor")
            from mlx_audio.stt.generate import load_model

            model = load_model(self._model_path)

            # mlx-community models don't include processor - load from original
            if not hasattr(model, "_processor") or model._processor is None:
                from transformers import WhisperProcessor

                # Map mlx-community models to their OpenAI originals
                processor_map = {
                    "mlx-community/whisper-turbo": "openai/whisper-large-v3-turbo",
                    "mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
                    "mlx-community/whisper-large-v3-mlx": "openai/whisper-large-v3",
                    "mlx-community/distil-whisper-large-v3": "distil-whisper/distil-large-v3",
                }
                processor_path = processor_map.get(self._model_path, "openai/whisper-large-v3")
                model._processor = WhisperProcessor.from_pretrained(processor_path)

            self._model = model

        return self._model

    def transcribe_stream(
        self, audio_capture: AudioCapture
    ) -> Generator[TranscriptionSegment, None, None]:
        """Transcribe audio from capture stream in real-time.

        Args:
            audio_capture: AudioCapture instance providing audio chunks

        Yields:
            TranscriptionSegment objects with transcribed text
        """
        model = self._ensure_model()
        audio_buffer: list[np.ndarray] = []

        for chunk in audio_capture.get_audio_stream():
            audio_buffer.append(chunk.data)

            # Convert accumulated audio to mx.array
            audio = mx.array(np.concatenate(audio_buffer), dtype=mx.float32)

            # Stream transcription
            for result in model.generate_streaming(
                audio,
                chunk_duration=self._chunk_duration,
                frame_threshold=self._frame_threshold,
                language=self._language,
            ):
                yield TranscriptionSegment(
                    text=result.text,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    is_final=result.is_final,
                )

    def transcribe_audio(
        self, audio_data: np.ndarray, audio_buffer: list[np.ndarray]
    ) -> Generator[TranscriptionSegment, None, None]:
        """Transcribe audio data directly (for main-thread processing).

        Args:
            audio_data: New audio chunk to add
            audio_buffer: Accumulated audio buffer (modified in-place, cleared on final)

        Yields:
            TranscriptionSegment objects with transcribed text
        """
        model = self._ensure_model()
        audio_buffer.append(audio_data)
        audio = mx.array(np.concatenate(audio_buffer), dtype=mx.float32)

        for result in model.generate_streaming(
            audio,
            chunk_duration=self._chunk_duration,
            frame_threshold=self._frame_threshold,
            language=self._language,
        ):
            segment = TranscriptionSegment(
                text=result.text,
                start_time=result.start_time,
                end_time=result.end_time,
                is_final=result.is_final,
            )
            yield segment
            # Clear buffer after final segment to prevent O(n²) re-processing
            if segment.is_final:
                audio_buffer.clear()
                return  # Exit after final to avoid stale buffer state

    def transcribe_batch(self, audio_buffer: list[np.ndarray]) -> TranscriptionSegment:
        """Transcribe accumulated audio in batch mode for maximum accuracy.

        Args:
            audio_buffer: Accumulated audio buffer

        Returns:
            Single TranscriptionSegment with full transcription
        """
        if not audio_buffer:
            return TranscriptionSegment(text="", start_time=0.0, end_time=0.0, is_final=True)

        model = self._ensure_model()
        audio = np.concatenate(audio_buffer)

        result = model.generate(
            audio,
            temperature=self._temperature,
            language=self._language,
            initial_prompt=self._initial_prompt,
            condition_on_previous_text=True,
        )

        return TranscriptionSegment(
            text=result.text.strip(),
            start_time=0.0,
            end_time=len(audio) / 16000,
            is_final=True,
        )

    def close(self) -> None:
        """Release model resources."""
        self._model = None

    def __enter__(self) -> "StreamingTranscriber":
        """Enter context manager, preloading model."""
        self._ensure_model()  # Preload so it's ready when user starts recording
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, releasing resources."""
        self.close()
