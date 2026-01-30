"""Pytest configuration for transcription tests.

Mocks external dependencies (sqlite_vec, sounddevice, mlx, etc.)
that may not be available in the test environment.
"""

import sys
from unittest.mock import MagicMock

# Mock sqlite_vec before any theo imports
mock_sqlite_vec = MagicMock()
sys.modules["sqlite_vec"] = mock_sqlite_vec

# Mock sounddevice for audio capture tests
mock_sd = MagicMock()
mock_sd.InputStream = MagicMock()
mock_sd.query_devices = MagicMock(return_value={"default_samplerate": 16000})
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
sys.modules["mlx"] = mock_mlx
sys.modules["mlx.core"] = mock_mlx_core
sys.modules["mlx_audio"] = mock_mlx_audio
sys.modules["mlx_audio.stt"] = mock_mlx_audio_stt
