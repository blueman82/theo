"""JSON protocol definitions for daemon IPC communication.

This module defines the message format for communication between the
daemon server and clients over Unix sockets.

Protocol Overview:
    - All messages are newline-delimited JSON
    - Requests: {"cmd": "...", "args": {...}, "request_id": "..."}
    - Responses: {"success": true/false, "data": {...}, "error": "...", "request_id": "..."}

Commands:
    - ping: Health check
    - embed: Generate embeddings for texts
    - status: Get daemon status
    - shutdown: Request graceful shutdown (admin only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

__all__ = [
    "DaemonCommand",
    "DaemonRequest",
    "DaemonResponse",
    "JobStatus",
    "EmbedJob",
]


class DaemonCommand(str, Enum):
    """Available daemon commands.

    Commands are grouped by function:
    - Core: ping, status, shutdown
    - Embedding: embed, embed_async, get_job_status
    - Index Operations: index, search, delete

    Note: index, search, and delete commands use the embedding subsystem
    internally for vector operations. They provide a higher-level interface
    for document management than raw embed commands.
    """

    # Core commands
    PING = "ping"
    STATUS = "status"
    SHUTDOWN = "shutdown"

    # Embedding commands
    EMBED = "embed"
    EMBED_ASYNC = "embed_async"
    GET_JOB_STATUS = "get_job_status"

    # Index operations (use embedding internally)
    INDEX = "index"
    SEARCH = "search"
    DELETE = "delete"


class JobStatus(str, Enum):
    """Status of an async embedding job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DaemonRequest:
    """Protocol message for daemon requests.

    All messages to the daemon follow this format.

    Attributes:
        cmd: Command name (ping, embed, embed_async, get_job_status, status, shutdown).
        args: Command arguments as dictionary.
        request_id: Optional request identifier for correlation.
    """

    cmd: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    @classmethod
    def from_json(cls, data: str | bytes) -> DaemonRequest | None:
        """Parse a DaemonRequest from JSON string.

        Args:
            data: JSON string or bytes to parse.

        Returns:
            Parsed DaemonRequest or None if invalid.

        Example:
            >>> req = DaemonRequest.from_json('{"cmd": "ping"}')
            >>> req.cmd
            'ping'
        """
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                return None

            cmd = parsed.get("cmd")
            if not cmd or not isinstance(cmd, str):
                return None

            return cls(
                cmd=cmd,
                args=parsed.get("args") or {},
                request_id=parsed.get("request_id"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def to_json(self) -> str:
        """Serialize to JSON string with newline.

        Returns:
            JSON-encoded request string with newline.
        """
        message = {
            "cmd": self.cmd,
            "args": self.args,
        }
        if self.request_id:
            message["request_id"] = self.request_id

        return json.dumps(message) + "\n"


@dataclass(slots=True)
class DaemonResponse:
    """Protocol message for daemon responses.

    Attributes:
        success: Whether the operation succeeded.
        data: Response data if successful.
        error: Error message if failed.
        request_id: Correlated request identifier.
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    request_id: str | None = None

    @classmethod
    def from_json(cls, data: str | bytes) -> DaemonResponse | None:
        """Parse a DaemonResponse from JSON string.

        Args:
            data: JSON string or bytes to parse.

        Returns:
            Parsed DaemonResponse or None if invalid.
        """
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                return None

            return cls(
                success=parsed.get("success", False),
                data=parsed.get("data"),
                error=parsed.get("error"),
                request_id=parsed.get("request_id"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def to_json(self) -> str:
        """Serialize to JSON string with newline.

        Returns:
            JSON-encoded response string with newline.
        """
        response: dict[str, Any] = {
            "success": self.success,
        }

        if self.data is not None:
            response["data"] = self.data

        if self.error is not None:
            response["error"] = self.error

        if self.request_id is not None:
            response["request_id"] = self.request_id

        return json.dumps(response) + "\n"

    @classmethod
    def ok(
        cls,
        data: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> DaemonResponse:
        """Create a successful response.

        Args:
            data: Response data.
            request_id: Correlated request ID.

        Returns:
            Success response.
        """
        return cls(success=True, data=data, request_id=request_id)

    @classmethod
    def err(
        cls,
        error: str,
        request_id: str | None = None,
    ) -> DaemonResponse:
        """Create an error response.

        Args:
            error: Error message.
            request_id: Correlated request ID.

        Returns:
            Error response.
        """
        return cls(success=False, error=error, request_id=request_id)


@dataclass(slots=True)
class EmbedJob:
    """Tracking structure for async embedding jobs.

    Attributes:
        job_id: Unique job identifier.
        texts: List of texts to embed.
        status: Current job status.
        embeddings: Resulting embeddings (when completed).
        error: Error message (when failed).
        created_at: When the job was created.
        completed_at: When the job completed (success or failure).
    """

    job_id: str
    texts: list[str]
    status: JobStatus = JobStatus.PENDING
    embeddings: list[list[float]] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the job.
        """
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status.value,
            "text_count": len(self.texts),
            "created_at": self.created_at.isoformat(),
        }

        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()

        if self.status == JobStatus.COMPLETED and self.embeddings:
            result["embedding_count"] = len(self.embeddings)
            result["embedding_dim"] = len(self.embeddings[0]) if self.embeddings else 0

        if self.status == JobStatus.FAILED and self.error:
            result["error"] = self.error

        return result

    def mark_processing(self) -> None:
        """Mark job as processing."""
        self.status = JobStatus.PROCESSING

    def mark_completed(self, embeddings: list[list[float]]) -> None:
        """Mark job as completed with results.

        Args:
            embeddings: The generated embeddings.
        """
        self.status = JobStatus.COMPLETED
        self.embeddings = embeddings
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark job as failed with error.

        Args:
            error: Error message describing the failure.
        """
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
