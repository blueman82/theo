"""Tests for streaming transcriber module.

Tests StreamingTranscriber with mocked mlx_audio dependencies.
"""

from unittest.mock import MagicMock, patch

import numpy as np


class TestStreamingTranscriber:
    """Tests for StreamingTranscriber class."""

    def test_streaming_transcriber_init(self) -> None:
        """Verify _model is None initially (lazy loading)."""
        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            transcriber = StreamingTranscriber()

            assert transcriber._model is None
            assert transcriber._model_path == "mlx-community/whisper-turbo"
            assert transcriber._chunk_duration == 1.0
            assert transcriber._frame_threshold == 25
            assert transcriber._language is None

    def test_streaming_transcriber_init_custom_params(self) -> None:
        """Verify custom parameters are accepted."""
        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            transcriber = StreamingTranscriber(
                model_path="custom/model",
                chunk_duration=0.5,
                frame_threshold=30,
                language="en",
            )

            assert transcriber._model_path == "custom/model"
            assert transcriber._chunk_duration == 0.5
            assert transcriber._frame_threshold == 30
            assert transcriber._language == "en"

    def test_streaming_transcriber_lazy_load(self) -> None:
        """Model loaded on _ensure_model(), not at init."""
        mock_model = MagicMock(name="mock_model")
        mock_model._processor = MagicMock()  # Simulate processor exists
        mock_load = MagicMock(return_value=mock_model)

        # Create mock mlx_audio.stt.generate module with load_model function
        mock_mlx_audio_stt_generate = MagicMock()
        mock_mlx_audio_stt_generate.load_model = mock_load

        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
                "mlx_audio.stt.generate": mock_mlx_audio_stt_generate,
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            transcriber = StreamingTranscriber()

            # Model not loaded yet
            assert transcriber._model is None

            # Call _ensure_model - it imports load_model from mlx_audio.stt.generate
            model = transcriber._ensure_model()

            # Model now loaded
            assert model is mock_model
            mock_load.assert_called_once_with("mlx-community/whisper-turbo")

    def test_streaming_transcriber_ensure_model_caches(self) -> None:
        """Verify _ensure_model caches the model after first load."""
        mock_model = MagicMock(name="mock_model")

        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            transcriber = StreamingTranscriber()
            transcriber._model = mock_model  # Simulate loaded model

            # Calls should return cached model
            model1 = transcriber._ensure_model()
            model2 = transcriber._ensure_model()

            assert model1 is model2
            assert model1 is mock_model

    def test_transcribe_stream_yields_segments(self) -> None:
        """Mock model.generate_streaming() to yield StreamingResult-like objects."""
        # Create mock streaming results
        class MockStreamingResult:
            def __init__(
                self, text: str, start: float, end: float, is_final: bool
            ) -> None:
                self.text = text
                self.start_time = start
                self.end_time = end
                self.is_final = is_final

        mock_results = [
            MockStreamingResult("Hello", 0.0, 1.0, True),
            MockStreamingResult("world", 1.0, 2.0, True),
        ]

        # Mock the model
        mock_model = MagicMock()
        mock_model.generate_streaming.return_value = iter(mock_results)

        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
            },
        ):
            from theo.transcription.types import AudioChunk
            from theo.transcription.transcriber import StreamingTranscriber

            # Mock AudioCapture
            mock_chunk = AudioChunk(
                data=np.zeros(16000, dtype=np.float32),
                timestamp=0.0,
            )

            mock_capture = MagicMock()
            mock_capture.get_audio_stream.return_value = iter([mock_chunk])

            # Create transcriber with mocked model
            transcriber = StreamingTranscriber()
            transcriber._model = mock_model

            segments = list(transcriber.transcribe_stream(mock_capture))

        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 1.0
        assert segments[0].is_final is True
        assert segments[1].text == "world"

    def test_transcriber_close(self) -> None:
        """Verify _model set to None after close()."""
        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            transcriber = StreamingTranscriber()
            transcriber._model = MagicMock(name="loaded_model")

            transcriber.close()

            assert transcriber._model is None

    def test_transcriber_context_manager(self) -> None:
        """Verify context manager calls close() on exit."""
        mock_model = MagicMock(name="mock_model")
        mock_model._processor = MagicMock()  # Simulate processor exists
        mock_load = MagicMock(return_value=mock_model)

        mock_mlx_audio_stt_generate = MagicMock()
        mock_mlx_audio_stt_generate.load_model = mock_load

        with patch.dict(
            "sys.modules",
            {
                "mlx": MagicMock(),
                "mlx.core": MagicMock(),
                "mlx_audio": MagicMock(),
                "mlx_audio.stt": MagicMock(),
                "mlx_audio.stt.generate": mock_mlx_audio_stt_generate,
            },
        ):
            from theo.transcription.transcriber import StreamingTranscriber

            with StreamingTranscriber() as transcriber:
                pass  # __enter__ loads model via _ensure_model

            # Model should be None after context exit
            assert transcriber._model is None
