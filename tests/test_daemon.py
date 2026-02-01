"""Tests for the daemon server and client.

Tests validate the daemon architecture for non-blocking operations:
- Protocol message parsing and serialization
- Job queue management
- Worker job processing
- Client connection and fallback
- Server request handling
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from theo.daemon.protocol import (
    DaemonCommand,
    DaemonRequest,
    DaemonResponse,
    EmbedJob,
    JobStatus,
)
from theo.daemon.worker import JobQueue, Worker


class TestDaemonRequest:
    """Test DaemonRequest protocol messages."""

    def test_from_json_valid(self):
        """Test parsing valid JSON request."""
        data = '{"cmd": "ping", "args": {"key": "value"}, "request_id": "abc123"}'
        req = DaemonRequest.from_json(data)

        assert req is not None
        assert req.cmd == "ping"
        assert req.args == {"key": "value"}
        assert req.request_id == "abc123"

    def test_from_json_minimal(self):
        """Test parsing minimal JSON request (cmd only)."""
        data = '{"cmd": "embed"}'
        req = DaemonRequest.from_json(data)

        assert req is not None
        assert req.cmd == "embed"
        assert req.args == {}
        assert req.request_id is None

    def test_from_json_bytes(self):
        """Test parsing bytes input."""
        data = b'{"cmd": "status"}'
        req = DaemonRequest.from_json(data)

        assert req is not None
        assert req.cmd == "status"

    def test_from_json_invalid(self):
        """Test parsing invalid JSON returns None."""
        assert DaemonRequest.from_json("not json") is None
        assert DaemonRequest.from_json("{}") is None  # No cmd
        assert DaemonRequest.from_json('{"cmd": 123}') is None  # cmd not string
        assert DaemonRequest.from_json('{"wrong": "format"}') is None

    def test_to_json(self):
        """Test serialization to JSON."""
        req = DaemonRequest(cmd="embed", args={"texts": ["hello"]}, request_id="xyz")
        json_str = req.to_json()

        assert json_str.endswith("\n")
        parsed = json.loads(json_str.strip())
        assert parsed["cmd"] == "embed"
        assert parsed["args"]["texts"] == ["hello"]
        assert parsed["request_id"] == "xyz"


class TestDaemonResponse:
    """Test DaemonResponse protocol messages."""

    def test_from_json_success(self):
        """Test parsing successful response."""
        data = '{"success": true, "data": {"embeddings": [[0.1, 0.2]]}, "request_id": "abc"}'
        resp = DaemonResponse.from_json(data)

        assert resp is not None
        assert resp.success is True
        assert resp.data == {"embeddings": [[0.1, 0.2]]}
        assert resp.error is None
        assert resp.request_id == "abc"

    def test_from_json_error(self):
        """Test parsing error response."""
        data = '{"success": false, "error": "Something went wrong"}'
        resp = DaemonResponse.from_json(data)

        assert resp is not None
        assert resp.success is False
        assert resp.error == "Something went wrong"

    def test_ok_factory(self):
        """Test ok() factory method."""
        resp = DaemonResponse.ok(data={"count": 5}, request_id="test")

        assert resp.success is True
        assert resp.data == {"count": 5}
        assert resp.error is None
        assert resp.request_id == "test"

    def test_err_factory(self):
        """Test err() factory method."""
        resp = DaemonResponse.err(error="Failed", request_id="test")

        assert resp.success is False
        assert resp.error == "Failed"
        assert resp.data is None
        assert resp.request_id == "test"

    def test_to_json(self):
        """Test serialization to JSON."""
        resp = DaemonResponse.ok(data={"result": "ok"})
        json_str = resp.to_json()

        assert json_str.endswith("\n")
        parsed = json.loads(json_str.strip())
        assert parsed["success"] is True
        assert parsed["data"]["result"] == "ok"


class TestEmbedJob:
    """Test EmbedJob tracking structure."""

    def test_creation(self):
        """Test job creation with defaults."""
        job = EmbedJob(job_id="test123", texts=["hello", "world"])

        assert job.job_id == "test123"
        assert job.texts == ["hello", "world"]
        assert job.status == JobStatus.PENDING
        assert job.embeddings is None
        assert job.error is None
        assert job.completed_at is None
        assert isinstance(job.created_at, datetime)

    def test_mark_processing(self):
        """Test marking job as processing."""
        job = EmbedJob(job_id="test", texts=["hello"])
        job.mark_processing()

        assert job.status == JobStatus.PROCESSING

    def test_mark_completed(self):
        """Test marking job as completed."""
        job = EmbedJob(job_id="test", texts=["hello"])
        embeddings = [[0.1, 0.2, 0.3]]
        job.mark_completed(embeddings)

        assert job.status == JobStatus.COMPLETED
        assert job.embeddings == embeddings
        assert job.completed_at is not None

    def test_mark_failed(self):
        """Test marking job as failed."""
        job = EmbedJob(job_id="test", texts=["hello"])
        job.mark_failed("Something broke")

        assert job.status == JobStatus.FAILED
        assert job.error == "Something broke"
        assert job.completed_at is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        job = EmbedJob(job_id="test", texts=["hello", "world"])
        d = job.to_dict()

        assert d["job_id"] == "test"
        assert d["status"] == "pending"
        assert d["text_count"] == 2
        assert "created_at" in d

    def test_to_dict_completed(self):
        """Test to_dict includes embedding info when completed."""
        job = EmbedJob(job_id="test", texts=["hello"])
        job.mark_completed([[0.1, 0.2, 0.3]])
        d = job.to_dict()

        assert d["status"] == "completed"
        assert d["embedding_count"] == 1
        assert d["embedding_dim"] == 3
        assert "completed_at" in d


class TestJobQueue:
    """Test JobQueue management."""

    def test_add_and_get_pending(self):
        """Test adding and retrieving pending jobs."""
        queue = JobQueue()

        job1 = EmbedJob(job_id="1", texts=["a"])
        job2 = EmbedJob(job_id="2", texts=["b"])

        queue.add(job1)
        queue.add(job2)

        assert queue.pending_count() == 2

        retrieved = queue.get_pending()
        assert retrieved is not None
        assert retrieved.job_id == "1"
        # Job is still pending in _jobs dict until processed
        # but removed from _pending deque
        # Mark as processing to change count
        retrieved.mark_processing()
        assert queue.pending_count() == 1

    def test_get_job_by_id(self):
        """Test retrieving job by ID."""
        queue = JobQueue()
        job = EmbedJob(job_id="abc", texts=["test"])
        queue.add(job)

        retrieved = queue.get_job("abc")
        assert retrieved is not None
        assert retrieved.job_id == "abc"

        assert queue.get_job("nonexistent") is None

    def test_stats(self):
        """Test queue statistics."""
        queue = JobQueue()

        job1 = EmbedJob(job_id="1", texts=["a"])
        job2 = EmbedJob(job_id="2", texts=["b"])
        job3 = EmbedJob(job_id="3", texts=["c"])

        queue.add(job1)
        queue.add(job2)
        queue.add(job3)

        # Process one job
        job = queue.get_pending()
        if job:
            job.mark_completed([[0.1]])

        stats = queue.stats()
        assert stats["pending"] == 2
        assert stats["completed"] == 1
        assert stats["total"] == 3

    def test_eviction_on_capacity(self):
        """Test that old completed jobs are evicted when at capacity."""
        queue = JobQueue(max_jobs=5)

        # Fill queue with completed jobs
        for i in range(5):
            job = EmbedJob(job_id=str(i), texts=["x"])
            job.mark_completed([[0.1]])
            queue.add(job)

        assert len(queue._jobs) == 5

        # Add one more - should trigger eviction
        new_job = EmbedJob(job_id="new", texts=["y"])
        queue.add(new_job)

        # Some old jobs should have been evicted
        assert len(queue._jobs) <= 5


class TestWorker:
    """Test Worker background processing."""

    def test_worker_initialization(self):
        """Test worker initialization."""
        mock_provider = Mock()
        mock_provider.embed_texts = Mock(return_value=[[0.1, 0.2]])

        queue = JobQueue()
        worker = Worker(mock_provider, queue, poll_interval=0.1)

        assert worker.provider == mock_provider
        assert worker.queue == queue
        assert worker.poll_interval == 0.1
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_worker_processes_job(self):
        """Test that worker processes jobs from queue."""
        # Create mock provider
        mock_provider = Mock()
        mock_provider.embed_texts = Mock(return_value=[[0.1, 0.2, 0.3]])

        queue = JobQueue()
        worker = Worker(mock_provider, queue, poll_interval=0.05)

        # Add a job to the queue
        job = EmbedJob(job_id="test", texts=["hello"])
        queue.add(job)

        # Start worker
        worker.start()

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop worker
        await worker.stop()

        # Verify job was processed
        assert job.status == JobStatus.COMPLETED
        assert job.embeddings == [[0.1, 0.2, 0.3]]
        assert worker._jobs_processed == 1

    @pytest.mark.asyncio
    async def test_worker_handles_errors(self):
        """Test that worker handles embedding errors gracefully."""
        # Create mock provider that fails
        mock_provider = Mock()
        mock_provider.embed_texts = Mock(side_effect=Exception("Embedding failed"))

        queue = JobQueue()
        worker = Worker(mock_provider, queue, poll_interval=0.05)

        # Add a job to the queue
        job = EmbedJob(job_id="test", texts=["hello"])
        queue.add(job)

        # Start worker
        worker.start()

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop worker
        await worker.stop()

        # Verify job was marked as failed
        assert job.status == JobStatus.FAILED
        assert job.error == "Embedding failed"

    @pytest.mark.asyncio
    async def test_worker_status(self):
        """Test worker status reporting."""
        mock_provider = Mock()
        queue = JobQueue()
        worker = Worker(mock_provider, queue, poll_interval=0.1)

        worker.start()
        await asyncio.sleep(0.1)

        status = worker.status()
        assert status["running"] is True
        assert status["jobs_processed"] == 0
        assert "queue" in status

        await worker.stop()

        status = worker.status()
        assert status["running"] is False


class TestDaemonClient:
    """Test DaemonClient functionality."""

    def test_client_initialization(self):
        """Test client initialization with defaults."""
        from theo.daemon.client import SOCKET_PATH, DaemonClient

        client = DaemonClient()

        assert client.socket_path == SOCKET_PATH
        assert client.auto_fallback is True
        assert client._connected is False

    def test_client_custom_socket_path(self):
        """Test client with custom socket path."""
        from theo.daemon.client import DaemonClient

        custom_path = Path("/tmp/custom.sock")
        client = DaemonClient(socket_path=custom_path)

        assert client.socket_path == custom_path

    def test_client_context_manager(self):
        """Test client as context manager."""
        from theo.daemon.client import DaemonClient

        with DaemonClient() as client:
            assert isinstance(client, DaemonClient)

        assert not client._connected

    def test_ping_fallback_mode(self):
        """Test ping in fallback mode (daemon not running)."""
        from theo.daemon.client import DaemonClient

        # Use non-existent socket path
        with tempfile.TemporaryDirectory() as tmpdir:
            client = DaemonClient(
                socket_path=Path(tmpdir) / "nonexistent.sock",
                auto_fallback=True,
            )

            result = client.ping()

            assert result["success"] is True
            assert result.get("fallback") is True
            assert result["data"]["pong"] is False

    def test_embed_fallback_mode(self):
        """Test embed falls back to direct embedding."""
        from theo.daemon.client import DaemonClient

        # Create mock embedding provider - patch where it's imported
        with patch("theo.embedding.create_embedding_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.embed_texts = Mock(return_value=[[0.1, 0.2, 0.3]])
            mock_provider.close = Mock()
            mock_create.return_value = mock_provider

            with tempfile.TemporaryDirectory() as tmpdir:
                client = DaemonClient(
                    socket_path=Path(tmpdir) / "nonexistent.sock",
                    auto_fallback=True,
                )

                result = client.embed(texts=["hello"])

                assert result["success"] is True
                assert result.get("fallback") is True
                assert result["data"]["embeddings"] == [[0.1, 0.2, 0.3]]
                mock_provider.close.assert_called_once()

    def test_is_daemon_running_false(self):
        """Test is_daemon_running returns False when daemon not running."""
        from theo.daemon.client import DaemonClient

        # Should return False since daemon is not running
        assert DaemonClient.is_daemon_running() is False


class TestDaemonServerHandlers:
    """Test DaemonServer request handlers."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock embedding provider."""
        provider = Mock()
        # embed_texts should return one embedding per input text
        provider.embed_texts = Mock(
            side_effect=lambda texts: [[0.1, 0.2, 0.3] * 341 for _ in texts]
        )
        provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        provider.health_check = Mock(return_value=True)
        provider.close = Mock()
        return provider

    @pytest.fixture
    def server(self, mock_provider):
        """Create a DaemonServer with mock provider."""
        from theo.daemon.server import DaemonServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = DaemonServer(
                socket_path=Path(tmpdir) / "test.sock",
                embedding_provider=mock_provider,
            )
            yield server

    @pytest.mark.asyncio
    async def test_handle_ping(self, server):
        """Test ping handler."""
        result = await server._handle_ping({})

        assert result["pong"] is True
        assert "timestamp" in result
        assert "pid" in result

    @pytest.mark.asyncio
    async def test_handle_embed(self, server, mock_provider):
        """Test synchronous embed handler."""
        result = await server._handle_embed({"texts": ["hello"]})

        assert "embeddings" in result
        assert result["count"] == 1
        mock_provider.embed_texts.assert_called_once_with(["hello"])

    @pytest.mark.asyncio
    async def test_handle_embed_missing_texts(self, server):
        """Test embed handler with missing texts argument."""
        with pytest.raises(ValueError, match="texts"):
            await server._handle_embed({})

    @pytest.mark.asyncio
    async def test_handle_embed_async(self, server):
        """Test async embed handler."""
        result = await server._handle_embed_async({"texts": ["hello", "world"]})

        assert result["queued"] is True
        assert "job_id" in result
        assert result["text_count"] == 2

        # Verify job was added to queue
        job = server.queue.get_job(result["job_id"])
        assert job is not None
        assert job.texts == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_handle_get_job_status(self, server):
        """Test get_job_status handler."""
        # Create and add a job
        job = EmbedJob(job_id="test123", texts=["hello"])
        server.queue.add(job)

        result = await server._handle_get_job_status({"job_id": "test123"})

        assert result["job_id"] == "test123"
        assert result["status"] == "pending"
        assert result["text_count"] == 1

    @pytest.mark.asyncio
    async def test_handle_get_job_status_not_found(self, server):
        """Test get_job_status with nonexistent job."""
        with pytest.raises(ValueError, match="not found"):
            await server._handle_get_job_status({"job_id": "nonexistent"})

    @pytest.mark.asyncio
    async def test_handle_status(self, server):
        """Test status handler."""
        result = await server._handle_status({})

        assert "pid" in result
        assert "socket_path" in result
        assert "embedding_provider" in result
        assert "worker" in result
        assert result["health_check"] is True

    @pytest.mark.asyncio
    async def test_handle_index(self, server, mock_provider):
        """Test index handler generates embeddings for documents."""
        result = await server._handle_index({"texts": ["doc1", "doc2"]})

        assert result["indexed"] is True
        assert "embeddings" in result
        assert len(result["ids"]) == 2
        assert result["count"] == 2
        mock_provider.embed_texts.assert_called_with(["doc1", "doc2"])

    @pytest.mark.asyncio
    async def test_handle_index_with_ids(self, server, mock_provider):
        """Test index handler with provided document IDs."""
        result = await server._handle_index(
            {
                "texts": ["doc1", "doc2"],
                "ids": ["id1", "id2"],
            }
        )

        assert result["indexed"] is True
        assert result["ids"] == ["id1", "id2"]
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_handle_index_mismatched_ids(self, server):
        """Test index handler rejects mismatched ids length."""
        with pytest.raises(ValueError, match="ids.*must match"):
            await server._handle_index(
                {
                    "texts": ["doc1", "doc2"],
                    "ids": ["id1"],  # Only one ID for two texts
                }
            )

    @pytest.mark.asyncio
    async def test_handle_index_missing_texts(self, server):
        """Test index handler with missing texts argument."""
        with pytest.raises(ValueError, match="texts"):
            await server._handle_index({})

    @pytest.mark.asyncio
    async def test_handle_search(self, server, mock_provider):
        """Test search handler generates query embedding."""
        mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])

        result = await server._handle_search({"query": "find documents"})

        assert result["query"] == "find documents"
        assert "embedding" in result
        assert result["dimensions"] == 3
        mock_provider.embed_query.assert_called_with("find documents")

    @pytest.mark.asyncio
    async def test_handle_search_missing_query(self, server):
        """Test search handler with missing query argument."""
        with pytest.raises(ValueError, match="query"):
            await server._handle_search({})

    @pytest.mark.asyncio
    async def test_handle_delete(self, server):
        """Test delete handler acknowledges deletion request."""
        result = await server._handle_delete({"ids": ["id1", "id2", "id3"]})

        assert result["acknowledged"] is True
        assert result["ids"] == ["id1", "id2", "id3"]
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_handle_delete_missing_ids(self, server):
        """Test delete handler with missing ids argument."""
        with pytest.raises(ValueError, match="ids"):
            await server._handle_delete({})


class TestDaemonCommand:
    """Test DaemonCommand enum."""

    def test_command_values(self):
        """Test command enum values."""
        # Core commands
        assert DaemonCommand.PING.value == "ping"
        assert DaemonCommand.STATUS.value == "status"
        assert DaemonCommand.SHUTDOWN.value == "shutdown"

        # Embedding commands
        assert DaemonCommand.EMBED.value == "embed"
        assert DaemonCommand.EMBED_ASYNC.value == "embed_async"
        assert DaemonCommand.GET_JOB_STATUS.value == "get_job_status"

        # Index operations
        assert DaemonCommand.INDEX.value == "index"
        assert DaemonCommand.SEARCH.value == "search"
        assert DaemonCommand.DELETE.value == "delete"


class TestJobStatus:
    """Test JobStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestQuickEmbed:
    """Test quick_embed convenience function."""

    def test_quick_embed_fallback(self):
        """Test quick_embed with fallback (when daemon not running)."""
        # The quick_embed function imports create_embedding_provider inside the
        # _do_fallback method, so patch needs to be on theo.embedding module
        with patch("theo.embedding.factory.create_embedding_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.embed_texts = Mock(return_value=[[0.1, 0.2]])
            mock_provider.close = Mock()
            mock_create.return_value = mock_provider

            from theo.daemon.client import quick_embed

            embeddings = quick_embed(["hello"])

            assert embeddings == [[0.1, 0.2]]


@pytest.mark.xdist_group("daemon")
class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_client(self):
        """Test get_client factory function."""
        from theo.daemon.client import get_client

        client = get_client(auto_fallback=False, connect_timeout=5.0)

        assert client.auto_fallback is False
        assert client.connect_timeout == 5.0

    def test_is_daemon_running(self):
        """Test is_daemon_running function returns a boolean."""
        from theo.daemon.server import is_daemon_running

        # Function should return a boolean (daemon might be running from hooks)
        result = is_daemon_running()
        assert isinstance(result, bool)


@pytest.mark.timeout(60)
@pytest.mark.xdist_group("daemon")
class TestIntegration:
    """Integration tests for daemon components."""

    @pytest.mark.asyncio
    async def test_full_async_embedding_workflow(self):
        """Test complete async embedding workflow."""
        from theo.daemon.server import DaemonServer

        # Create mock provider
        mock_provider = Mock()
        mock_provider.embed_texts = Mock(return_value=[[0.1, 0.2, 0.3]])
        mock_provider.health_check = Mock(return_value=True)
        mock_provider.close = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            server = DaemonServer(
                socket_path=Path(tmpdir) / "test.sock",
                embedding_provider=mock_provider,
            )

            # Start worker
            server.worker.start()

            try:
                # Queue async job
                result = await server._handle_embed_async({"texts": ["test text"]})
                job_id = result["job_id"]

                # Wait for processing
                await asyncio.sleep(0.3)

                # Check status
                status = await server._handle_get_job_status({"job_id": job_id})

                assert status["status"] == "completed"
                assert "embeddings" in status
                assert status["embeddings"] == [[0.1, 0.2, 0.3]]

            finally:
                await server.worker.stop()
