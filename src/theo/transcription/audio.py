"""Audio capture for Theo transcription.

Provides microphone audio capture using sounddevice with callback-based streaming.
Audio chunks are queued for consumption by transcription pipeline.
"""

import logging
import queue
import time
from collections.abc import Generator
from types import TracebackType
from typing import cast

import numpy as np
import sounddevice as sd
from scipy import signal

from theo.transcription.types import AudioChunk

logger = logging.getLogger(__name__)


class AudioCapture:
    """Microphone audio capture with callback-based streaming.

    Captures audio from the default (or specified) input device and yields
    AudioChunk objects suitable for Whisper transcription (16kHz mono).

    Example:
        with AudioCapture() as capture:
            capture.start()
            for chunk in capture.get_audio_stream():
                # Process chunk
                pass
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 1.0,
        device: int | None = None,
    ) -> None:
        """Initialize audio capture.

        Args:
            sample_rate: Target sample rate (16000 Hz for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration per audio chunk in seconds
            device: Input device index, or None for default
        """
        self._target_sample_rate = sample_rate
        self._channels = channels
        self._chunk_duration = chunk_duration
        self._device = device
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[AudioChunk | None] = queue.Queue()
        self._recording = False
        self._start_time: float | None = None
        self._device_sample_rate: int | None = None
        self._device_name: str = "Unknown"

    @property
    def is_recording(self) -> bool:
        """Return True if currently recording."""
        return self._recording

    @property
    def device_name(self) -> str:
        """Return the name of the active input device."""
        return self._device_name

    def start(self) -> None:
        """Start audio capture.

        Creates an InputStream and begins capturing audio via callback.
        Audio is resampled to target sample rate if device rate differs.
        """
        if self._recording:
            return

        # Get device info to determine actual sample rate
        device_info = sd.query_devices(self._device, kind="input")
        self._device_sample_rate = int(device_info["default_samplerate"])
        self._device_name = str(device_info.get("name", "Unknown"))

        # Calculate blocksize based on device's native rate
        blocksize = int(self._device_sample_rate * self._chunk_duration)

        self._stream = sd.InputStream(
            samplerate=self._device_sample_rate,
            channels=self._channels,
            blocksize=blocksize,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._recording = True
        self._start_time = time.time()
        logger.info(
            "Started audio capture: device_rate=%d, target_rate=%d",
            self._device_sample_rate,
            self._target_sample_rate,
        )

    def stop(self) -> None:
        """Stop audio capture and clean up resources."""
        if not self._recording:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._recording = False
        self._queue.put(None)  # Sentinel to signal end
        logger.info("Stopped audio capture")

    def get_audio_stream(self) -> Generator[AudioChunk, None, None]:
        """Yield audio chunks from the capture queue.

        Yields:
            AudioChunk objects with audio data and timestamps

        Note:
            Generator terminates when stop() is called (receives None sentinel).
        """
        while True:
            try:
                chunk = self._queue.get(timeout=2.0)
            except queue.Empty:
                if not self._recording:
                    break
                continue
            if chunk is None:
                break
            yield chunk

    def get_chunk_nowait(self) -> AudioChunk | None:
        """Get next audio chunk without blocking.

        Returns:
            AudioChunk if available, None if queue is empty or stop sentinel received.

        Raises:
            queue.Empty: If no chunk is immediately available.
        """
        chunk = self._queue.get_nowait()
        return chunk

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Process incoming audio data from sounddevice.

        Resamples audio to target sample rate if needed and queues AudioChunk.
        """
        if status:
            logger.warning("Audio callback status: %s", status)

        # Copy data (indata is a buffer that will be reused)
        audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

        # Skip silent chunks (prevents Whisper hallucinations like "Thank you")
        rms_energy = np.sqrt(np.mean(audio_data**2))
        if rms_energy < 0.01:  # Threshold for silence
            return

        # Resample if device rate differs from target
        device_rate = self._device_sample_rate or self._target_sample_rate
        if device_rate != self._target_sample_rate:
            target_samples = int(len(audio_data) * self._target_sample_rate / device_rate)
            resampled = cast(np.ndarray, signal.resample(audio_data, target_samples))
            audio_data = resampled.astype(np.float32)

        timestamp = time.time() - self._start_time if self._start_time else 0.0

        chunk = AudioChunk(
            data=audio_data,
            timestamp=timestamp,
            sample_rate=self._target_sample_rate,
        )

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk at timestamp=%.2f", timestamp)

    def __enter__(self) -> "AudioCapture":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, ensuring capture is stopped."""
        if self._recording:
            self.stop()
