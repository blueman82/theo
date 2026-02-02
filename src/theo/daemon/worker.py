"""Background worker for processing embedding jobs.

This module provides the Worker class that processes embedding jobs
asynchronously in the background. It integrates with theo.embedding
providers to generate embeddings without blocking the main server.

Architecture:
    Worker runs as asyncio.create_task in daemon lifecycle
    Polls job queue -> Generates embeddings -> Updates job status
    Graceful handling when embedding provider is unavailable
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from theo.daemon.protocol import EmbedJob, JobStatus

if TYPE_CHECKING:
    from theo.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

__all__ = ["Worker", "JobQueue"]


class JobQueue:
    """Thread-safe queue for embedding jobs.

    Manages pending jobs and provides job status lookups.

    Attributes:
        _pending: Deque of jobs waiting to be processed.
        _jobs: Dictionary of all jobs by ID for status lookup.
        _max_jobs: Maximum number of jobs to track (for memory bounds).
    """

    def __init__(self, max_jobs: int = 1000) -> None:
        """Initialize the job queue.

        Args:
            max_jobs: Maximum jobs to track before evicting oldest completed.
        """
        self._pending: deque[EmbedJob] = deque()
        self._jobs: dict[str, EmbedJob] = {}
        self._max_jobs = max_jobs

    def add(self, job: EmbedJob) -> None:
        """Add a job to the queue.

        Args:
            job: The embedding job to queue.
        """
        # Evict oldest completed jobs if at capacity
        if len(self._jobs) >= self._max_jobs:
            self._evict_completed()

        self._jobs[job.job_id] = job
        self._pending.append(job)

    def get_pending(self) -> EmbedJob | None:
        """Get the next pending job.

        Returns:
            Next pending job or None if queue is empty.
        """
        while self._pending:
            job = self._pending.popleft()
            # Skip jobs that were already processed (shouldn't happen)
            if job.status == JobStatus.PENDING:
                return job
        return None

    def get_job(self, job_id: str) -> EmbedJob | None:
        """Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The job or None if not found.
        """
        return self._jobs.get(job_id)

    def pending_count(self) -> int:
        """Get count of pending jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)

    def processing_count(self) -> int:
        """Get count of processing jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)

    def completed_count(self) -> int:
        """Get count of completed jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)

    def failed_count(self) -> int:
        """Get count of failed jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)

    def stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dictionary with job counts by status.
        """
        return {
            "pending": self.pending_count(),
            "processing": self.processing_count(),
            "completed": self.completed_count(),
            "failed": self.failed_count(),
            "total": len(self._jobs),
        }

    def _evict_completed(self) -> None:
        """Evict oldest completed jobs to make room."""
        # Find completed jobs sorted by completion time
        completed = [
            (j.completed_at or j.created_at, j.job_id)
            for j in self._jobs.values()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]
        completed.sort()

        # Remove oldest half
        to_remove = len(completed) // 2
        for _, job_id in completed[:to_remove]:
            del self._jobs[job_id]


class Worker:
    """Background worker for processing embedding jobs.

    The worker runs as an async task, polling the job queue for pending
    jobs and processing them using the configured embedding provider.

    Attributes:
        provider: The embedding provider to use.
        queue: The job queue to process.
        poll_interval: Seconds to sleep when queue is empty.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        queue: JobQueue,
        poll_interval: float = 0.5,
    ) -> None:
        """Initialize the worker.

        Args:
            provider: Embedding provider for generating embeddings.
            queue: Job queue to process.
            poll_interval: Seconds to sleep when no work available.
        """
        self.provider = provider
        self.queue = queue
        self.poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._jobs_processed = 0
        self._start_time: datetime | None = None

    async def _process_job(self, job: EmbedJob) -> None:
        """Process a single embedding job.

        Args:
            job: The job to process.
        """
        job.mark_processing()

        try:
            # Run embedding in executor to not block event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.provider.embed_texts,
                job.texts,
            )
            job.mark_completed(embeddings)
            self._jobs_processed += 1

        except Exception as e:
            job.mark_failed(str(e))
            logger.warning(f"Job {job.job_id} failed: {e}")

    async def _worker_loop(self) -> None:
        """Main worker loop that processes jobs."""
        self._running = True
        self._start_time = datetime.now()
        logger.info(f"Worker started (poll_interval={self.poll_interval}s)")

        while self._running:
            try:
                # Get next pending job
                job = self.queue.get_pending()

                if job is None:
                    # No work, sleep and retry
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Process the job
                await self._process_job(job)

                # Clear caches to prevent memory accumulation
                gc.collect()
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except ImportError:
                    pass  # MLX not available

            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info("Worker stopped")

    def start(self) -> None:
        """Start the worker as a background task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> dict[str, object]:
        """Get worker status.

        Returns:
            Dictionary with worker status information.
        """
        return {
            "running": self._running,
            "jobs_processed": self._jobs_processed,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "poll_interval": self.poll_interval,
            "queue": self.queue.stats(),
        }
