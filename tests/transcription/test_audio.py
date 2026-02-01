"""Tests for audio capture module.

Tests AudioCapture with mocked sounddevice dependencies.
Module mocks are set up in conftest.py.
"""

from unittest.mock import MagicMock

import numpy as np

from theo.transcription.audio import AudioCapture


class TestAudioCapture:
    """Tests for AudioCapture class."""

    def test_audio_capture_init(self) -> None:
        """Verify default sample_rate=16000 and channels=1."""
        capture = AudioCapture()

        assert capture._target_sample_rate == 16000
        assert capture._channels == 1
        assert capture._chunk_duration == 1.0
        assert capture._device is None

    def test_audio_capture_init_custom_params(self) -> None:
        """Verify custom parameters are accepted."""
        capture = AudioCapture(
            sample_rate=44100,
            channels=2,
            chunk_duration=0.5,
            device=1,
        )

        assert capture._target_sample_rate == 44100
        assert capture._channels == 2
        assert capture._chunk_duration == 0.5
        assert capture._device == 1

    def test_audio_capture_start_stop(self, mocker: MagicMock) -> None:
        """Mock sounddevice.InputStream, verify state transitions."""
        mock_sd = mocker.patch("theo.transcription.audio.sd")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"default_samplerate": 48000}

        capture = AudioCapture()

        # Initially not recording
        assert capture.is_recording is False

        # Start capture
        capture.start()

        assert capture.is_recording is True
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

        # Stop capture
        capture.stop()

        assert capture.is_recording is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_audio_capture_start_idempotent(self, mocker: MagicMock) -> None:
        """Starting when already recording should be idempotent."""
        mock_sd = mocker.patch("theo.transcription.audio.sd")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"default_samplerate": 48000}

        capture = AudioCapture()
        capture.start()
        capture.start()  # Second start should do nothing

        # InputStream should only be created once
        assert mock_sd.InputStream.call_count == 1

    def test_audio_capture_stop_idempotent(self, mocker: MagicMock) -> None:
        """Stopping when not recording should be idempotent."""
        mocker.patch("theo.transcription.audio.sd")

        capture = AudioCapture()
        capture.stop()  # Should not raise

        assert capture.is_recording is False

    def test_audio_capture_context_manager(self, mocker: MagicMock) -> None:
        """Verify __enter__ and __exit__ work correctly."""
        mock_sd = mocker.patch("theo.transcription.audio.sd")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"default_samplerate": 48000}

        with AudioCapture() as capture:
            capture.start()
            assert capture.is_recording is True

        # After exiting context, capture should be stopped
        assert capture.is_recording is False

    def test_audio_capture_is_recording_property(self, mocker: MagicMock) -> None:
        """Property returns correct state throughout lifecycle."""
        mock_sd = mocker.patch("theo.transcription.audio.sd")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"default_samplerate": 48000}

        capture = AudioCapture()

        # Initial state
        assert capture.is_recording is False

        # After start
        capture.start()
        assert capture.is_recording is True

        # After stop
        capture.stop()
        assert capture.is_recording is False


class TestAudioCaptureCallback:
    """Tests for audio callback processing."""

    def test_audio_callback_queues_chunk(self, mocker: MagicMock) -> None:
        """Verify callback queues AudioChunk correctly."""
        mock_sd = mocker.patch("theo.transcription.audio.sd")
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.query_devices.return_value = {"default_samplerate": 16000, "name": "Mock Device"}

        capture = AudioCapture()
        capture.start()

        # Simulate callback with non-silent audio data (must exceed RMS threshold of 0.01)
        indata = np.full((16000, 1), 0.1, dtype=np.float32)
        mock_status = MagicMock()
        mock_status.__bool__ = MagicMock(return_value=False)

        capture._audio_callback(indata, 16000, {}, mock_status)

        # Check that chunk was queued
        chunk = capture._queue.get_nowait()
        assert chunk is not None
        assert len(chunk.data) == 16000
        assert chunk.sample_rate == 16000
