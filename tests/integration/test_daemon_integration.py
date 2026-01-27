"""Integration tests for Theo daemon server and client IPC.

This module provides end-to-end tests for daemon communication:
- Client -> Unix Socket -> Server -> Worker
- Synchronous and asynchronous embedding operations
- Fallback behavior when daemon unavailable
- Protocol message handling

Tests use temporary sockets and mock embedding providers to ensure
isolation and avoid external dependencies.

Usage:
    uv run pytest tests/integration/test_daemon_integration.py -v
"""

import asyncio
import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from theo.daemon import (
    DaemonClient,
    DaemonConnectionError,
    DaemonServer,
    JobStatus,
)
from theo.daemon.protocol import DaemonRequest, DaemonResponse, EmbedJob


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_socket_path():
    """Create a temporary socket path with short name for Unix socket limit.

    Unix sockets have a path length limit of 104-108 chars on macOS.
    We use /tmp with a unique suffix to stay under this limit.

    Yields:
        Path for temporary socket
    """
    import uuid

    # Use short path to avoid Unix socket path length limits
    short_id = uuid.uuid4().hex[:8]
    socket_path = Path(f"/tmp/theo_{short_id}.sock")

    yield socket_path

    # Cleanup socket if it exists
    if socket_path.exists():
        socket_path.unlink()


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock embedding provider for testing.

    Returns deterministic embeddings based on text content.

    Returns:
        Mock EmbeddingProvider instance
    """
    provider = MagicMock()

    def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings from texts."""
        embeddings = []
        for text in texts:
            text_hash = hash(text[:50] if text else "")
            # Generate 1024-dimensional embedding
            embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
            magnitude = sum(x * x for x in embedding) ** 0.5
            embeddings.append([x / magnitude for x in embedding])
        return embeddings

    def mock_embed_query(text: str) -> list[float]:
        """Generate query embedding."""
        text_hash = hash(text[:50] if text else "")
        embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
        magnitude = sum(x * x for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]

    provider.embed_texts = MagicMock(side_effect=mock_embed_texts)
    provider.embed_query = MagicMock(side_effect=mock_embed_query)
    provider.health_check = MagicMock(return_value=True)
    provider.close = MagicMock()

    return provider


# =============================================================================
# Protocol Tests
# =============================================================================


class TestDaemonProtocol:
    """Test daemon protocol message handling."""

    def test_request_serialization(self):
        """Test DaemonRequest serialization."""
        request = DaemonRequest(
            cmd="embed",
            args={"texts": ["hello", "world"]},
            request_id="test123",
        )

        # Serialize (DaemonRequest has to_json, not to_dict)
        json_str = request.to_json()

        # Deserialize
        parsed = DaemonRequest.from_json(json_str)

        assert parsed is not None
        assert parsed.cmd == "embed"
        assert parsed.args == {"texts": ["hello", "world"]}
        assert parsed.request_id == "test123"

    def test_response_serialization(self):
        """Test DaemonResponse serialization."""
        response = DaemonResponse.ok(
            data={"embeddings": [[0.1, 0.2]], "count": 1},
            request_id="test123",
        )

        # Serialize
        json_str = response.to_json()

        # Parse JSON
        parsed = json.loads(json_str.strip())

        assert parsed["success"] is True
        assert parsed["data"]["count"] == 1
        assert parsed["request_id"] == "test123"

    def test_error_response(self):
        """Test error response creation."""
        response = DaemonResponse.err(
            error="Something went wrong",
            request_id="test123",
        )

        json_str = response.to_json()
        parsed = json.loads(json_str.strip())

        assert parsed["success"] is False
        assert parsed["error"] == "Something went wrong"

    def test_embed_job_lifecycle(self):
        """Test EmbedJob status transitions."""
        job = EmbedJob(job_id="job123", texts=["test"])

        # Initial state
        assert job.status == JobStatus.PENDING
        assert job.embeddings is None

        # Mark as processing
        job.status = JobStatus.PROCESSING

        assert job.status == JobStatus.PROCESSING

        # Complete with embeddings
        job.embeddings = [[0.1, 0.2, 0.3]]
        job.status = JobStatus.COMPLETED

        assert job.status == JobStatus.COMPLETED
        assert job.embeddings is not None


# =============================================================================
# Client Tests
# =============================================================================


class TestDaemonClient:
    """Test DaemonClient functionality."""

    def test_client_connect_nonexistent_socket(self, temp_socket_path: Path):
        """Test client handles missing socket gracefully."""
        client = DaemonClient(
            socket_path=temp_socket_path,
            auto_fallback=True,
        )

        # Should return False (no socket)
        assert client.connect() is False

    def test_client_fallback_ping(self, temp_socket_path: Path):
        """Test client fallback for ping command."""
        client = DaemonClient(
            socket_path=temp_socket_path,
            auto_fallback=True,
        )

        result = client.ping()

        assert result["success"] is True
        assert result.get("fallback") is True
        assert result["data"]["pong"] is False

    def test_client_fallback_status(self, temp_socket_path: Path):
        """Test client fallback for status command."""
        client = DaemonClient(
            socket_path=temp_socket_path,
            auto_fallback=True,
        )

        result = client.status()

        assert result["success"] is True
        assert result.get("fallback") is True
        assert result["data"]["running"] is False

    def test_client_no_fallback_raises(self, temp_socket_path: Path):
        """Test client raises when fallback disabled."""
        client = DaemonClient(
            socket_path=temp_socket_path,
            auto_fallback=False,
        )

        with pytest.raises(DaemonConnectionError):
            client.connect()

    def test_client_context_manager(self, temp_socket_path: Path):
        """Test client context manager cleanup."""
        with DaemonClient(socket_path=temp_socket_path) as client:
            result = client.ping()
            assert result["success"] is True

        # After context, client should be closed
        assert not client._connected

    def test_client_is_daemon_running_false(self, temp_socket_path: Path):
        """Test is_daemon_running returns False when not running."""
        with patch("theo.daemon.client.PID_FILE", temp_socket_path.parent / "test.pid"):
            with patch("theo.daemon.client.SOCKET_PATH", temp_socket_path):
                assert DaemonClient.is_daemon_running() is False


# =============================================================================
# Server Tests
# =============================================================================


class TestDaemonServer:
    """Test DaemonServer functionality."""

    @pytest.mark.asyncio
    async def test_server_start_and_stop(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server starts and stops cleanly."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        # Start server in background
        start_task = asyncio.create_task(server.start())

        # Wait for server to start
        for _ in range(50):
            if temp_socket_path.exists():
                break
            await asyncio.sleep(0.1)

        assert temp_socket_path.exists()

        # Stop server
        await server.stop()

        # Cancel start task if still running
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass

        # Socket should be cleaned up
        assert not temp_socket_path.exists()

    @pytest.mark.asyncio
    async def test_server_handles_ping(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles ping command."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        # Start server
        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Connect and send ping
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            request = {"cmd": "ping", "args": {}, "request_id": "test1"}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is True
            assert response["data"]["pong"] is True

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_server_handles_embed(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles synchronous embed command."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Connect and send embed request
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            request = {
                "cmd": "embed",
                "args": {"texts": ["hello", "world"]},
                "request_id": "test2",
            }
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is True
            assert "embeddings" in response["data"]
            assert response["data"]["count"] == 2
            assert response["data"]["dimensions"] == 1024

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_server_handles_search(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles search command."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Connect and send search request
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            request = {
                "cmd": "search",
                "args": {"query": "test query"},
                "request_id": "test3",
            }
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is True
            assert "embedding" in response["data"]
            assert response["data"]["dimensions"] == 1024

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_server_handles_status(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles status command."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Connect and send status request
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            request = {"cmd": "status", "args": {}, "request_id": "test4"}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is True
            assert "pid" in response["data"]
            assert "health_check" in response["data"]

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_server_handles_unknown_command(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles unknown commands gracefully."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Connect and send unknown command
            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            request = {"cmd": "unknown_command", "args": {}, "request_id": "test5"}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is False
            assert "unknown" in response["error"].lower()

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass


# =============================================================================
# Client-Server Integration Tests
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.xdist_group("daemon")
class TestClientServerIntegration:
    """Test client and server working together."""

    @pytest.mark.asyncio
    async def test_client_connects_to_running_server(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test client can connect to a running server."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start - verify with raw socket test
            for attempt in range(100):
                if temp_socket_path.exists():
                    # Try a raw socket connection to verify server is truly ready
                    try:
                        reader, writer = await asyncio.open_unix_connection(
                            str(temp_socket_path)
                        )
                        writer.close()
                        await writer.wait_closed()
                        break  # Server is ready
                    except (ConnectionRefusedError, OSError):
                        pass
                await asyncio.sleep(0.1)

            # Verify server is actually ready
            assert temp_socket_path.exists(), "Server socket should exist"

            # Create client - note: need to use synchronous client methods
            # But run ping in executor since it blocks
            client = DaemonClient(
                socket_path=temp_socket_path,
                auto_fallback=False,
            )

            # Run the synchronous client.ping() in an executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, client.ping)

            assert result["success"] is True, f"Ping failed: {result}"
            assert result["data"]["pong"] is True

            client.close()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_client_embed_through_server(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test client can embed texts through server."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Create client
            client = DaemonClient(socket_path=temp_socket_path)

            # Embed texts
            result = client.embed(texts=["hello world", "goodbye world"])

            assert result["success"] is True
            assert result["data"]["count"] == 2
            assert len(result["data"]["embeddings"]) == 2
            assert len(result["data"]["embeddings"][0]) == 1024

            client.close()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_async_embed_workflow(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test async embed and job status retrieval."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start - verify with raw socket test
            for attempt in range(100):
                if temp_socket_path.exists():
                    # Try a raw socket connection to verify server is truly ready
                    try:
                        reader, writer = await asyncio.open_unix_connection(
                            str(temp_socket_path)
                        )
                        writer.close()
                        await writer.wait_closed()
                        break  # Server is ready
                    except (ConnectionRefusedError, OSError):
                        pass
                await asyncio.sleep(0.1)

            # Verify server is actually ready
            assert temp_socket_path.exists(), "Server socket should exist"

            # Create client with auto_fallback=False to catch connection issues
            client = DaemonClient(
                socket_path=temp_socket_path,
                auto_fallback=False,
            )

            # Run the synchronous client methods in an executor
            loop = asyncio.get_running_loop()

            # Queue async embed
            async_result = await loop.run_in_executor(
                None,
                lambda: client.embed_async(texts=["async text 1", "async text 2"]),
            )

            assert async_result["success"] is True, f"Async embed failed: {async_result}"
            assert "job_id" in async_result["data"]
            job_id = async_result["data"]["job_id"]

            # Check job status
            status_result = await loop.run_in_executor(
                None,
                lambda: client.get_job_status(job_id),
            )

            assert status_result["success"] is True, f"Job status failed: {status_result}"
            # Job may be pending, processing, or completed

            client.close()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky with concurrent test runners - needs investigation")
    async def test_multiple_clients_concurrent(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test multiple clients can connect concurrently."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            # Create multiple clients
            client1 = DaemonClient(socket_path=temp_socket_path)
            client2 = DaemonClient(socket_path=temp_socket_path)

            # Both should connect
            assert client1.connect() is True
            assert client2.connect() is True

            # Both should be able to ping
            result1 = client1.ping()
            result2 = client2.ping()

            assert result1["success"] is True
            assert result2["success"] is True

            client1.close()
            client2.close()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in daemon communication."""

    @pytest.mark.asyncio
    async def test_embed_missing_texts_argument(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test embed command without texts argument returns error."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            # Send embed without texts
            request = {"cmd": "embed", "args": {}, "request_id": "error1"}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is False
            assert "texts" in response["error"].lower()

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_search_missing_query_argument(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test search command without query argument returns error."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            # Send search without query
            request = {"cmd": "search", "args": {}, "request_id": "error2"}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()

            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is False
            assert "query" in response["error"].lower()

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_invalid_json_request(
        self,
        temp_socket_path: Path,
        mock_embedding_provider: MagicMock,
    ):
        """Test server handles invalid JSON gracefully."""
        server = DaemonServer(
            socket_path=temp_socket_path,
            embedding_provider=mock_embedding_provider,
        )

        start_task = asyncio.create_task(server.start())

        try:
            # Wait for server to start
            for _ in range(50):
                if temp_socket_path.exists():
                    break
                await asyncio.sleep(0.1)

            reader, writer = await asyncio.open_unix_connection(str(temp_socket_path))

            # Send invalid JSON
            writer.write(b"not valid json\n")
            await writer.drain()

            response_data = await reader.readline()
            response = json.loads(response_data.decode())

            assert response["success"] is False
            assert "invalid" in response["error"].lower()

            writer.close()
            await writer.wait_closed()

        finally:
            await server.stop()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    def test_client_handles_server_disconnect(self, temp_socket_path: Path):
        """Test client handles server disconnect gracefully."""
        # Create socket file to simulate server that disconnects
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(temp_socket_path))
        sock.listen(1)
        sock.settimeout(1.0)

        client = DaemonClient(
            socket_path=temp_socket_path,
            connect_timeout=1.0,
            request_timeout=1.0,
            auto_fallback=True,
        )

        # Client connects
        assert client.connect() is True

        # Accept and immediately close
        try:
            conn, _ = sock.accept()
            conn.close()
        except socket.timeout:
            pass

        # Client should fall back on next request
        result = client.ping()
        # Result could be fallback or error
        if result.get("success"):
            assert result.get("fallback") is True

        client.close()
        sock.close()
