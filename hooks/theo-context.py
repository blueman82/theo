#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "pydantic",
#     "pydantic-settings",
# ]
# ///
"""Claude Code / Factory SessionStart hook for loading relevant memory context.

This hook runs at the start of each session and injects relevant memories
as system context. It uses Ollama (gemma3:12b) for intelligent curation
and synthesis of memories.

SessionStart matchers: startup, resume, clear, compact

Architecture:
    When daemon is running (fast path):
        - Uses DaemonClient IPC for fetch and curate operations
        - Daemon keeps Ollama warm and caches memory lookups

    When daemon is unavailable (fallback path):
        - Falls back to subprocess calls for theo
        - Falls back to direct Ollama subprocess for curation

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or
    ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-context.py",
                            "timeout": 10
                        }
                    ]
                }
            ]
        }
    }

The hook outputs markdown context that the agent will see at session start.
Failures are handled gracefully - they don't block the agent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from theo_client import DaemonClient, get_shared_client

if TYPE_CHECKING:
    from typing import Any

# Note: Using get_shared_client() from theo_client.py for connection reuse


def _is_daemon_available() -> bool:
    """Check if daemon is running and available.

    Returns:
        True if daemon is available for IPC.
    """
    return DaemonClient.is_daemon_running()

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LLM_MODEL = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

THEO_TIMEOUT_SECONDS = 5
OLLAMA_CURATE_TIMEOUT_SECONDS = 8

PROJECT_INDICATORS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)

# Include all meaningful memory types except "session" (which is tool tracking noise)
GLOBAL_MEMORY_TYPES = frozenset({"preference", "golden_rule", "pattern", "decision", "workflow"})

MEMORY_TYPE_CATEGORIES = (
    "golden_rule",
    "preference",
    "pattern",
    "decision",
)

RFC_2119_PREAMBLE = (
    'The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", '
    '"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in these '
    "memories are to be interpreted as described in RFC 2119."
)

MEMORY_TIP = (
    "**Memory Tools Available:** `memory_store` (store new), "
    "`memory_recall` (search), `memory_validate` (adjust confidence), "
    "`memory_apply`/`memory_outcome` (TRY-LEARN cycle), "
    "`memory_relate` (link memories), `memory_forget` (delete). "
    'Use via mcp-exec with wrappers: ["theo"]'
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Immutable context about the current project.

    Attributes:
        namespace (e.g., 'project:theo'' or 'global').
        name: The project directory name.
        root: Absolute path to the project root.
    """

    namespace: str
    name: str
    root: Path


@dataclass(slots=True)
class Memory:
    """A single memory entry with metadata.

    Attributes:
        id: Unique identifier for the memory.
        type: Memory type (preference, pattern, decision, etc.).
        content: The actual memory content.
        importance: Importance score (0.0 to 1.0).
        confidence: Confidence score (0.0 to 1.0).
        source: Where the memory came from (project, global, etc.).
        via_graph: Whether this memory was found via graph expansion.
        metadata: Additional metadata dict.
        relevance: Relevance score for graph-expanded memories.
        path: Graph traversal path for expanded memories.
    """

    id: str
    type: str
    content: str
    importance: float = 0.5
    confidence: float = 0.3
    source: str = "unknown"
    via_graph: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance: float | None = None
    path: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        source: str = "unknown",
        via_graph: bool = False,
    ) -> Memory | None:
        """Create a Memory from a raw dictionary.

        Args:
            data: Raw memory dictionary from theo API.
            source: The source label for this memory.
            via_graph: Whether this was found via graph expansion.

        Returns:
            A Memory instance, or None if the data is invalid.
        """
        if not data or not isinstance(data, dict):
            return None

        memory_id = data.get("id")
        if not memory_id:
            return None

        return cls(
            id=memory_id,
            type=data.get("type", "unknown"),
            content=data.get("content", ""),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.3),
            source=source,
            via_graph=via_graph,
            metadata=data.get("metadata") or {},
            relevance=data.get("relevance_score"),
            path=data.get("path", []),
        )

    @property
    def project_path(self) -> str:
        """Get the project path from metadata, if any."""
        return self.metadata.get("project_path", "")


# =============================================================================
# Ollama Integration (Subprocess Fallback)
# =============================================================================


def _run_ollama_curate_subprocess(prompt: str, model: str) -> str | None:
    """Run Ollama via subprocess (fallback when daemon unavailable).

    This is only used when the daemon is not running. When daemon is
    available, curate_with_ollama() uses client.send('curate', ...).

    Args:
        prompt: The full curation prompt.
        model: The Ollama model to use.

    Returns:
        The curated output, or None on failure.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "run", model, "--think=medium", "--hidethinking"],
            check=False, input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_CURATE_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            return None

        output = result.stdout.strip()

        if not output or "Memory Context" not in output:
            return None

        return output

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


# =============================================================================
# Project Detection
# =============================================================================


def get_project_context() -> ProjectContext:
    """Derive project namespace and name from current working directory.

    Walks up the directory tree looking for common project indicators
    (.git, pyproject.toml, etc.) to find the project root.

    Returns:
        A ProjectContext with namespace, name, and root path.
    """
    import os

    cwd = Path(os.getcwd())
    project_name = cwd.name
    project_root = cwd

    # Walk up to find project root
    current = cwd
    while current != current.parent:
        for indicator in PROJECT_INDICATORS:
            if (current / indicator).exists():
                return ProjectContext(
                    namespace=f"project:{current.name}",
                    name=current.name,
                    root=current,
                )
        current = current.parent

    # Check current directory as fallback
    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return ProjectContext(
                namespace=f"project:{project_name}",
                name=project_name,
                root=cwd,
            )

    return ProjectContext(
        namespace="global",
        name=project_name,
        root=cwd,
    )


# =============================================================================
# Recall Integration
# =============================================================================


def _find_theo_directory() -> Path | None:
    """Find the theo module directory.

    Returns:
        Path to the theo directory, or None if not found.
    """
    theo_paths = [
        Path.home() / "Documents" / "Github" / "theo",
        Path(__file__).parent.parent,
        Path.home() / "Github" / "theo",
        Path.home() / ".local" / "share" / "theo",
        Path("/opt/theo"),
    ]

    for path in theo_paths:
        if (path / "src" / "theo" / "__main__.py").exists():
            return path

    return None


def _call_theo_subprocess(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call theo MCP tool via subprocess (fallback when daemon unavailable).

    This is only used when the daemon is not running. When daemon is
    available, fetch_raw_memories() uses client.send('fetch', ...).

    Args:
        tool_name: Name of the tool (memory_list_tool, etc.).
        args: Dictionary of tool arguments.

    Returns:
        Tool result as dictionary, or error dict on failure.
    """
    import json
    import subprocess

    try:
        theo_dir = _find_theo_directory()

        if theo_dir is None:
            cmd = [
                "uv", "run", "python", "-m", "theo",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]
            working_dir = Path.cwd()
        else:
            cmd = [
                "uv", "run",
                "--directory", str(theo_dir),
                "python", "-m", "theo",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]
            working_dir = theo_dir

        result = subprocess.run(
            cmd,
            check=False, capture_output=True,
            text=True,
            timeout=THEO_TIMEOUT_SECONDS,
            cwd=working_dir,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"theo failed: {result.stderr}",
            }

        parsed = json.loads(result.stdout)
        if parsed is None:
            return {"success": False, "error": "theo returned null"}

        return parsed

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "theo timed out"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON response: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": "uv or python not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Memory Fetching
# =============================================================================


def _is_relevant_to_project(memory: Memory, project_root: Path) -> bool:
    """Check if a memory is relevant to the current project.

    Filters out memories that have a different project path stored.

    Args:
        memory: The memory to check.
        project_root: The current project's root path.

    Returns:
        True if the memory is relevant, False otherwise.
    """
    mem_project = memory.project_path

    # No project path = global memory, always relevant
    if not mem_project:
        return True

    # Same project path = relevant
    if mem_project == str(project_root):
        return True

    # Global preferences/rules apply everywhere
    if memory.type in GLOBAL_MEMORY_TYPES:
        return True

    # Different project's decisions/patterns/workflows = not relevant
    return False


def _process_memories_from_result(
    result: dict[str, Any],
    seen_ids: set[str],
    project_root: Path,
    source: str,
    include_expanded: bool = True,
    type_filter: frozenset[str] | None = None,
    min_relevance: float = 0.0,
) -> list[Memory]:
    """Process memories from a theo result.

    Args:
        result: The raw result from theo API (daemon or subprocess).
        seen_ids: Set of already-seen memory IDs (modified in-place).
        project_root: The current project's root path.
        source: Source label for these memories.
        include_expanded: Whether to include graph-expanded memories.
        type_filter: If set, only include memories of these types.
        min_relevance: Minimum relevance score for expanded memories.

    Returns:
        List of processed Memory objects.
    """
    memories: list[Memory] = []

    if not result.get("success"):
        return memories

    # Process primary memories
    for mem_data in result.get("memories") or []:
        memory = Memory.from_dict(mem_data, source=source, via_graph=False)
        if memory is None:
            continue
        if memory.id in seen_ids:
            continue
        if type_filter and memory.type not in type_filter:
            continue
        if not _is_relevant_to_project(memory, project_root):
            continue

        seen_ids.add(memory.id)
        memories.append(memory)

    # Process expanded memories
    if not include_expanded:
        return memories

    expanded_source = f"{source} (via graph)"
    for exp_data in result.get("expanded") or []:
        memory = Memory.from_dict(exp_data, source=expanded_source, via_graph=True)
        if memory is None:
            continue
        if memory.id in seen_ids:
            continue
        if type_filter and memory.type not in type_filter:
            continue

        relevance = memory.relevance or 0.5
        if relevance <= min_relevance:
            continue
        if not _is_relevant_to_project(memory, project_root):
            continue

        seen_ids.add(memory.id)
        memories.append(memory)

    return memories


def _fetch_memories_via_daemon(
    client: DaemonClient,
    context: ProjectContext,
) -> list[Memory]:
    """Fetch memories via daemon IPC (fast path).

    Uses client.send('fetch', ...) for each memory query.

    Args:
        client: Connected DaemonClient instance.
        context: The current project context.

    Returns:
        List of Memory objects.
    """
    all_memories: list[Memory] = []
    seen_ids: set[str] = set()

    # Phase 1: Semantic search with graph expansion for project memories
    project_result = client.send(
        "fetch",
        query=(
            f"{context.name} project context preferences "
            "patterns decisions workflows"
        ),
        namespace=context.namespace,
        n_results=15,
        include_related=True,
        max_depth=1,
    )

    # Map daemon response format to expected format
    if project_result.get("success"):
        mapped = {
            "success": True,
            "memories": project_result.get("data", {}).get("memories", []),
            "expanded": project_result.get("data", {}).get("expanded", []),
        }
    else:
        mapped = {"success": False}

    all_memories.extend(_process_memories_from_result(
        mapped,
        seen_ids,
        context.root,
        source="project",
        include_expanded=True,
        min_relevance=0.6,
    ))

    # Phase 2: Global memories with graph expansion (preferences and golden rules)
    global_result = client.send(
        "fetch",
        query="user preferences coding style golden rules requirements",
        namespace="global",
        n_results=15,
        include_related=True,
        max_depth=1,
    )

    if global_result.get("success"):
        mapped_global = {
            "success": True,
            "memories": global_result.get("data", {}).get("memories", []),
            "expanded": global_result.get("data", {}).get("expanded", []),
        }
    else:
        mapped_global = {"success": False}

    all_memories.extend(_process_memories_from_result(
        mapped_global,
        seen_ids,
        context.root,
        source="global",
        include_expanded=True,
        type_filter=GLOBAL_MEMORY_TYPES,
    ))

    return all_memories


def _fetch_memories_via_subprocess(context: ProjectContext) -> list[Memory]:
    """Fetch memories via subprocess (fallback when daemon unavailable).

    Uses _call_theo_subprocess() for each memory query.

    Args:
        context: The current project context.

    Returns:
        List of Memory objects.
    """
    all_memories: list[Memory] = []
    seen_ids: set[str] = set()

    # Phase 1: Semantic search with graph expansion for project memories
    project_result = _call_theo_subprocess("memory_recall", {
        "query": (
            f"{context.name} project context preferences "
            "patterns decisions workflows"
        ),
        "namespace": context.namespace,
        "n_results": 15,
        "include_related": True,
        "max_depth": 1,
        "max_expanded": 15,
        "decay_factor": 0.8,
    })

    all_memories.extend(_process_memories_from_result(
        project_result,
        seen_ids,
        context.root,
        source="project",
        include_expanded=True,
        min_relevance=0.6,
    ))

    # Phase 2: High-importance project memories that might not match query
    project_list_result = _call_theo_subprocess("memory_list", {
        "namespace": context.namespace,
        "limit": 15,
        "order_by": "importance",
        "descending": True,
    })

    all_memories.extend(_process_memories_from_result(
        project_list_result,
        seen_ids,
        context.root,
        source="project",
        include_expanded=False,
    ))

    # Phase 3: Global memories with graph expansion (preferences and golden rules)
    global_result = _call_theo_subprocess("memory_recall", {
        "query": "user preferences coding style golden rules requirements",
        "namespace": "global",
        "n_results": 15,
        "include_related": True,
        "max_depth": 1,
        "max_expanded": 15,
    })

    all_memories.extend(_process_memories_from_result(
        global_result,
        seen_ids,
        context.root,
        source="global",
        include_expanded=True,
        type_filter=GLOBAL_MEMORY_TYPES,
    ))

    return all_memories


def fetch_raw_memories(context: ProjectContext) -> list[Memory]:
    """Fetch raw memories using semantic search with graph expansion.

    Uses DaemonClient.send('fetch', ...) when daemon is available,
    falling back to subprocess calls when daemon is unavailable.

    Args:
        context: The current project context.

    Returns:
        List of Memory objects with source and graph metadata.
    """
    # Fast path: Use daemon IPC if available
    if _is_daemon_available():
        client = get_shared_client()
        try:
            return _fetch_memories_via_daemon(client, context)
        except Exception:
            # Fall through to subprocess on any daemon error
            pass

    # Fallback: Use subprocess calls
    return _fetch_memories_via_subprocess(context)


# =============================================================================
# Context Generation
# =============================================================================


def _format_memories_for_curation(
    memories: list[Memory],
    context: ProjectContext,
) -> list[dict[str, Any]]:
    """Format Memory objects for daemon curate command.

    Args:
        memories: List of Memory objects.
        context: The current project context.

    Returns:
        List of memory dicts suitable for daemon curate command.
    """
    formatted = []
    for mem in memories:
        formatted.append({
            "id": mem.id,
            "type": mem.type,
            "content": mem.content,
            "importance": mem.importance,
            "confidence": mem.confidence,
            "source": mem.source,
            "metadata": mem.metadata,
        })
    return formatted


def _curate_via_daemon(
    client: DaemonClient,
    memories: list[Memory],
    context: ProjectContext,
    model: str,
) -> str | None:
    """Curate memories via daemon IPC (fast path).

    Uses client.send('curate', ...) for Ollama curation.

    Args:
        client: Connected DaemonClient instance.
        memories: List of Memory objects.
        context: The current project context.
        model: Ollama model to use.

    Returns:
        Curated markdown context, or None on failure.
    """
    formatted_memories = _format_memories_for_curation(memories, context)

    result = client.send(
        "curate",
        memories=formatted_memories,
        project_name=context.name,
        project_root=str(context.root),
        model=model,
    )

    if not result.get("success"):
        return None

    data = result.get("data", {})
    curated = data.get("curated") or data.get("context", "")

    if not curated or "Memory Context" not in curated:
        return None

    return curated


def _curate_via_subprocess(
    memories: list[Memory],
    context: ProjectContext,
    model: str,
) -> str | None:
    """Curate memories via subprocess (fallback when daemon unavailable).

    Uses _run_ollama_curate_subprocess() for Ollama curation.

    Args:
        memories: List of Memory objects.
        context: The current project context.
        model: Ollama model to use.

    Returns:
        Curated markdown context, or None on failure.
    """
    # Format memories for Ollama
    memory_lines: list[str] = []
    for mem in memories:
        project_tag = ""
        if mem.project_path:
            project_tag = f"|proj:{Path(mem.project_path).name}"

        line = (
            f"[{mem.source}|{mem.type}|imp:{mem.importance:.1f}"
            f"|conf:{mem.confidence:.1f}{project_tag}] {mem.content}"
        )
        memory_lines.append(line)

    memory_text = "\n".join(memory_lines)

    prompt = f"""You are curating memories for a Claude Code session.
Current project: {context.name}
Current path: {context.root}

Raw memories (format: [source|type|importance|confidence|project] content):
{memory_text}

CRITICAL INSTRUCTIONS:
1. PRESERVE exact RFC 2119 keywords from source (MUST, MUST NOT, SHOULD, etc.) - do NOT paraphrase or weaken them
2. PRESERVE important adverbs like "PROACTIVELY" - copy them exactly
3. Synthesize only TRUE duplicates (same meaning) - different rules are NOT duplicates
4. DO NOT remove rules just because of confidence differences - only remove if truly redundant
5. DO NOT add notes, explanations, or commentary - output ONLY the markdown sections
6. EXCLUDE memories clearly for different projects (mentions other project names, different tech stacks)
7. Keep memories relevant to "{context.name}" or truly global preferences

OUTPUT FORMAT:
# Memory Context

{RFC_2119_PREAMBLE}

---

## Golden Rules
- [highest priority rules, if any]

## Preferences
- [user preferences]

## Patterns
- [coding patterns]

## Recent Decisions
- [decisions, if any]

OUTPUT:"""

    return _run_ollama_curate_subprocess(prompt, model)


def curate_with_ollama(
    memories: list[Memory],
    context: ProjectContext,
    model: str = DEFAULT_LLM_MODEL,
) -> str | None:
    """Use Ollama to intelligently curate and synthesize memories.

    Uses DaemonClient.send('curate', ...) when daemon is available,
    falling back to direct Ollama subprocess when daemon is unavailable.

    Args:
        memories: List of Memory objects.
        context: The current project context.
        model: Ollama model to use (default: gemma3:12b).

    Returns:
        Curated markdown context, or None on failure.
    """
    if not memories:
        return None

    # Fast path: Use daemon IPC if available
    if _is_daemon_available():
        client = get_shared_client()
        try:
            result = _curate_via_daemon(client, memories, context, model)
            if result:
                return result
            # Fall through to subprocess if daemon curate failed
        except Exception:
            # Fall through to subprocess on any daemon error
            pass

    # Fallback: Use direct Ollama subprocess
    return _curate_via_subprocess(memories, context, model)


def fallback_context(memories: list[Memory]) -> str:
    """Generate simple context when Ollama is unavailable.

    Args:
        memories: List of Memory objects.

    Returns:
        Basic markdown context.
    """
    if not memories:
        return ""

    lines = [
        "# Memory Context",
        "",
        RFC_2119_PREAMBLE,
        "",
        "---",
        "",
    ]

    # Group by type
    by_type: dict[str, list[str]] = {cat: [] for cat in MEMORY_TYPE_CATEGORIES}
    by_type["other"] = []

    for mem in memories:
        namespace_tag = f" [{mem.source}]" if mem.source else ""
        formatted = f"- {mem.content}{namespace_tag}"

        if mem.type in by_type:
            by_type[mem.type].append(formatted)
        else:
            by_type["other"].append(formatted)

    section_headers = {
        "golden_rule": "## Golden Rules",
        "preference": "## Preferences",
        "pattern": "## Patterns",
        "decision": "## Recent Decisions",
    }

    for mem_type in MEMORY_TYPE_CATEGORIES:
        if by_type[mem_type]:
            lines.append(section_headers[mem_type])
            lines.extend(by_type[mem_type])
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# Logging
# =============================================================================


class Logger:
    """Simple file logger for hook debugging.

    Attributes:
        path: Path to the log file.
    """

    def __init__(self, log_dir: Path, filename: str = "theo-context.log") -> None:
        """Initialize the logger.

        Args:
            log_dir: Directory for log files.
            filename: Name of the log file.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / filename

    def log(self, message: str) -> None:
        """Write a timestamped message to the log file.

        Args:
            message: The message to log.
        """
        from datetime import datetime

        timestamp = datetime.now().isoformat()
        with self.path.open("a") as f:
            f.write(f"{timestamp} | {message}\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def _output_context(output: str, logger: Logger) -> None:
    """Output the curated context to stdout.

    Args:
        output: The markdown context to output.
        logger: Logger instance for debugging.
    """
    if output and output.strip():
        logger.log("outputting plain text context to stdout")
        print(output)
        print()
        print("---")
        print(MEMORY_TIP)
        logger.log("done")


def main() -> None:
    """Main hook entry point.

    Architecture:
        Both fetch_raw_memories() and curate_with_ollama() now internally
        use DaemonClient.send('fetch'/'curate', ...) when daemon is
        available, and fall back to subprocess calls when not.

        No warmup_ollama_model() needed - daemon handles Ollama keep-warm.

    All errors are caught to prevent blocking Claude Code.
    """
    log_dir = Path.home() / ".claude" / "hooks" / "logs"
    logger = Logger(log_dir)

    logger.log("SessionStart hook triggered")

    try:
        # Determine project context
        context = get_project_context()
        logger.log(
            f"namespace={context.namespace} "
            f"project={context.name} "
            f"root={context.root}",
        )

        # Log daemon status
        if _is_daemon_available():
            logger.log("daemon available, will use IPC fast path")
        else:
            logger.log("daemon not running, will use subprocess fallback")

        # Phase 1: Fetch raw memories
        # (uses daemon IPC if available, subprocess fallback otherwise)
        memories = fetch_raw_memories(context)
        logger.log(f"fetched {len(memories)} memories (after project filtering)")

        if not memories:
            logger.log("no memories, exiting")
            return

        # Phase 2: Curate with Ollama
        # (uses daemon IPC if available, subprocess fallback otherwise)
        output = curate_with_ollama(memories, context)
        logger.log(f"ollama curated: {len(output) if output else 0} chars")

        # Fallback if Ollama fails
        if not output:
            output = fallback_context(memories)
            logger.log(f"fallback context: {len(output) if output else 0} chars")

        _output_context(output, logger)

    except Exception as e:
        import traceback

        logger.log(f"ERROR: {e}")
        logger.log(f"TRACEBACK: {traceback.format_exc()}")
        print(f"<!-- theo-context hook error: {e} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
