"""Pytest configuration for transcription tests.

Mocks external dependencies (sqlite_vec, sounddevice, mlx, etc.)
that may not be available in the test environment.
"""

import sys
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

# Mock sqlite_vec before any theo imports
mock_sqlite_vec = MagicMock()
sys.modules["sqlite_vec"] = mock_sqlite_vec

# Mock sounddevice for audio capture tests
mock_sd = MagicMock()
mock_sd.InputStream = MagicMock()
mock_sd.query_devices = MagicMock(return_value={"default_samplerate": 16000, "name": "Mock Device"})
sys.modules["sounddevice"] = mock_sd

# Mock scipy.signal for resampling
mock_scipy = MagicMock()
mock_scipy_signal = MagicMock()
sys.modules["scipy"] = mock_scipy
sys.modules["scipy.signal"] = mock_scipy_signal

# Mock mlx for transcriber tests
mock_mlx = MagicMock()
mock_mlx_core = MagicMock()
mock_mlx_audio = MagicMock()
mock_mlx_audio_stt = MagicMock()
mock_mlx_audio_stt_generate = MagicMock()
sys.modules["mlx"] = mock_mlx
sys.modules["mlx.core"] = mock_mlx_core
sys.modules["mlx_audio"] = mock_mlx_audio
sys.modules["mlx_audio.stt"] = mock_mlx_audio_stt
sys.modules["mlx_audio.stt.generate"] = mock_mlx_audio_stt_generate


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Provide a mock Whisper model with load_model function."""
    mock_model = MagicMock(name="mock_whisper_model")
    mock_model._processor = MagicMock()
    mock_load = MagicMock(return_value=mock_model)
    mock_mlx_audio_stt_generate.load_model = mock_load
    yield mock_model
    # Reset after test
    mock_mlx_audio_stt_generate.load_model = MagicMock()


@pytest.fixture
def mock_sounddevice() -> dict[str, Any]:
    """Provide access to the mocked sounddevice module."""
    return {
        "sd": mock_sd,
        "InputStream": mock_sd.InputStream,
        "query_devices": mock_sd.query_devices,
    }
