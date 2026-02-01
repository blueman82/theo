"""Pytest configuration for transcription tests.

Mocks external dependencies (sounddevice, mlx, etc.)
that may not be available in the test environment.

Uses session-scoped autouse fixtures to properly isolate mocks
and prevent pollution of other test modules.
"""

import sys
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

# Store original sys.modules entries for cleanup
_original_modules: dict[str, Any] = {}

# Module names to mock for transcription tests
_MOCK_MODULES = [
    "sounddevice",
    "scipy",
    "scipy.signal",
    "mlx",
    "mlx.core",
    "mlx_audio",
    "mlx_audio.stt",
    "mlx_audio.stt.generate",
]


def _create_mocks() -> dict[str, MagicMock]:
    """Create all mock modules."""
    mock_sd = MagicMock()
    mock_sd.InputStream = MagicMock()
    mock_sd.query_devices = MagicMock(
        return_value={"default_samplerate": 16000, "name": "Mock Device"}
    )

    mock_scipy = MagicMock()
    mock_scipy_signal = MagicMock()

    mock_mlx = MagicMock()
    mock_mlx_core = MagicMock()
    mock_mlx_audio = MagicMock()
    mock_mlx_audio_stt = MagicMock()
    mock_mlx_audio_stt_generate = MagicMock()

    return {
        "sounddevice": mock_sd,
        "scipy": mock_scipy,
        "scipy.signal": mock_scipy_signal,
        "mlx": mock_mlx,
        "mlx.core": mock_mlx_core,
        "mlx_audio": mock_mlx_audio,
        "mlx_audio.stt": mock_mlx_audio_stt,
        "mlx_audio.stt.generate": mock_mlx_audio_stt_generate,
    }


# Create mocks once at module level for reuse in fixtures
_mocks = _create_mocks()


@pytest.fixture(autouse=True, scope="package")
def _install_transcription_mocks() -> Generator[None, None, None]:
    """Install mocks for transcription package and clean up after.

    Uses package scope so mocks are installed once per test package
    and cleaned up when the package tests complete.
    """
    global _original_modules

    # Save original modules and install mocks
    for name in _MOCK_MODULES:
        _original_modules[name] = sys.modules.get(name)
        sys.modules[name] = _mocks[name]

    yield

    # Restore original modules
    for name in _MOCK_MODULES:
        original = _original_modules.get(name)
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Provide a mock Whisper model with load_model function."""
    mock_model = MagicMock(name="mock_whisper_model")
    mock_model._processor = MagicMock()
    mock_load = MagicMock(return_value=mock_model)
    _mocks["mlx_audio.stt.generate"].load_model = mock_load
    yield mock_model
    # Reset after test
    _mocks["mlx_audio.stt.generate"].load_model = MagicMock()


@pytest.fixture
def mock_sounddevice() -> dict[str, MagicMock]:
    """Provide access to the mocked sounddevice module."""
    sd = _mocks["sounddevice"]
    return {
        "sd": sd,
        "InputStream": sd.InputStream,
        "query_devices": sd.query_devices,
    }
