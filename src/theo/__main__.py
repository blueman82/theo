"""MCP server entry point for Theo.

This module provides the main entry point for the Theo MCP server with:
- CLI argument parsing for flexible configuration
- Component initialization in dependency order
- Tool registration for all operations
- Signal handling for graceful shutdown
- Comprehensive logging to stderr (CRITICAL for MCP stdio)

Theo is a unified AI memory and document retrieval system that combines:
- DocVec: Semantic document indexing and retrieval
- Recall: Long-term memory with validation and confidence scoring

Usage:
    python -m theo [options]

    Options:
        --embedding-backend     Embedding backend: 'mlx' or 'ollama' (default: mlx)
        --mlx-model MODEL       MLX embedding model (default: mlx-community/mxbai-embed-large-v1)
        --ollama-host HOST      Ollama server host (default: http://localhost:11434)
        --ollama-model MODEL    Ollama embedding model (default: nomic-embed-text)
        --log-level LEVEL       Logging level (default: INFO)

CRITICAL: MCP servers using stdio transport must NEVER write to stdout
as it corrupts JSON-RPC messages. All logging goes to stderr.
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env file - must be done before any config access
load_dotenv()

# Initialize logging first, before any imports that might log
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Get required environment variable or raise error.

    Args:
        name: Environment variable name.

    Returns:
        The environment variable value.

    Raises:
        SystemExit: If variable is not set.
    """
    value = os.environ.get(name)
    if value is None:
        sys.stderr.write(f"ERROR: Required environment variable {name} not set.\n")
        sys.stderr.write("Please ensure .env file exists with all required variables.\n")
        sys.exit(1)
    return value


def setup_logging(log_level: str) -> None:
    """Configure logging to stderr (never stdout for MCP servers).

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Note:
        STDIO-based MCP servers must never write to stdout as it corrupts
        JSON-RPC messages. All logging goes to stderr.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger to write to stderr
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Critical: never use stdout in MCP servers
    )

    logger.info(f"Logging initialized at {log_level.upper()} level")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments from environment variables.

    All configuration comes from .env file (loaded via python-dotenv).
    No fallback defaults - all THEO_* variables must be set (except for trace subcommands).

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If any required environment variable is not set.
    """
    # Check for trace subcommand BEFORE argparse to avoid parse_known_args issues
    # (argparse would interpret unknown flags' values as positional subcommand args)
    if len(sys.argv) > 1 and sys.argv[1] == "trace":
        parser = argparse.ArgumentParser(
            description="Theo MCP server for unified AI memory and document retrieval",
        )
        subparsers = parser.add_subparsers(dest="subcommand")
        trace_parser = subparsers.add_parser("trace", help="Agent Trace commands")
        trace_subparsers = trace_parser.add_subparsers(dest="trace_command")
        trace_query_parser = trace_subparsers.add_parser("query", help="Query AI attribution for code")
        trace_query_parser.add_argument("file", type=str, help="File to query")
        trace_query_parser.add_argument(
            "--line", "-L", type=int, default=None, help="Line number to query"
        )
        return parser.parse_args()

    # MCP server mode - parse all arguments
    parser = argparse.ArgumentParser(
        description="Theo MCP server for unified AI memory and document retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Direct tool call mode (for daemon subprocess calls)
    parser.add_argument(
        "--call",
        type=str,
        metavar="TOOL_NAME",
        help="Call a tool directly and exit (for daemon use)",
    )
    parser.add_argument(
        "--args",
        type=str,
        default="{}",
        help="JSON arguments for --call mode",
    )

    # Embedding backend configuration (required from .env)
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default=_require_env("THEO_EMBEDDING_BACKEND"),
        choices=["mlx", "ollama"],
        help="Embedding backend (from THEO_EMBEDDING_BACKEND)",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=_require_env("THEO_MLX_MODEL"),
        help="MLX embedding model (from THEO_MLX_MODEL)",
    )

    # Ollama configuration (required from .env)
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=_require_env("THEO_OLLAMA_HOST"),
        help="Ollama server host (from THEO_OLLAMA_HOST)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=_require_env("THEO_OLLAMA_MODEL"),
        help="Ollama embedding model (from THEO_OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=int(_require_env("THEO_OLLAMA_TIMEOUT")),
        help="Ollama request timeout (from THEO_OLLAMA_TIMEOUT)",
    )

    # Logging configuration (required from .env)
    parser.add_argument(
        "--log-level",
        type=str,
        default=_require_env("THEO_LOG_LEVEL"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (from THEO_LOG_LEVEL)",
    )

    return parser.parse_args()


def initialize_components(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize all components in dependency order.

    Initialization order follows dependency graph:
    1. Embedding provider (MLX or Ollama)
    2. SQLiteStore (single source of truth for all storage)
    3. DaemonClient (connects to embedding provider)
    4. ChunkerRegistry
    5. HybridStore (wraps SQLiteStore with embedding generation)
    6. ValidationLoop
    7. Tool instances (IndexingTools, QueryTools, MemoryTools, ManagementTools)

    Args:
        args: Parsed CLI arguments

    Returns:
        Dictionary containing all initialized components

    Raises:
        Exception: If any component initialization fails
    """
    logger.info("Initializing components...")
    components: dict[str, Any] = {}

    try:
        # 1. Initialize embedding provider (MLX or Ollama based on config)
        logger.info(f"Initializing embedding provider (backend={args.embedding_backend})")
        from theo.embedding import create_embedding_provider

        embedder = create_embedding_provider(
            backend=args.embedding_backend,
            host=args.ollama_host,
            model=args.ollama_model,
            timeout=args.ollama_timeout,
            mlx_model=args.mlx_model,
        )
        components["embedder"] = embedder

        # 2. Initialize SQLiteStore (single source of truth for all storage)
        logger.info("Initializing SQLiteStore")
        from theo.config import TheoSettings
        from theo.storage.sqlite_store import SQLiteStore

        settings = TheoSettings()
        sqlite_store = SQLiteStore(db_path=settings.get_sqlite_path())
        components["sqlite_store"] = sqlite_store

        # 3. Initialize DaemonClient for non-blocking embedding operations
        # DaemonClient connects to daemon service or falls back to direct embedding
        logger.info("Initializing DaemonClient")
        from theo.daemon import DaemonClient

        daemon_client = DaemonClient(auto_fallback=True)
        components["daemon_client"] = daemon_client

        # 4. Initialize ChunkerRegistry
        logger.info("Initializing ChunkerRegistry")
        from theo.chunking import ChunkerRegistry

        chunker_registry = ChunkerRegistry()
        components["chunker_registry"] = chunker_registry

        # 5. Initialize HybridStore (wraps SQLiteStore with embedding generation)
        logger.info("Initializing HybridStore")
        from theo.storage.hybrid import HybridStore

        hybrid_store = HybridStore(
            sqlite_store=sqlite_store,
            embedding_client=embedder,
        )
        components["hybrid_store"] = hybrid_store

        # 6. Initialize ValidationLoop
        logger.info("Initializing ValidationLoop")
        from theo.validation import ValidationLoop

        validation_loop = ValidationLoop(store=sqlite_store)
        components["validation_loop"] = validation_loop

        # 7. Initialize FeedbackCollector for search feedback
        logger.info("Initializing FeedbackCollector")
        from theo.validation import FeedbackCollector

        feedback_collector = FeedbackCollector()
        components["feedback_collector"] = feedback_collector

        # 8. Initialize Tool instances
        logger.info("Initializing IndexingTools")
        from theo.tools import IndexingTools

        indexing_tools = IndexingTools(
            daemon_client=daemon_client,
            chunker_registry=chunker_registry,
            store=sqlite_store,
        )
        components["indexing_tools"] = indexing_tools

        logger.info("Initializing QueryTools")
        from theo.tools import QueryTools

        query_tools = QueryTools(
            daemon_client=daemon_client,
            store=sqlite_store,
            feedback_collector=feedback_collector,
        )
        components["query_tools"] = query_tools

        logger.info("Initializing MemoryTools")
        from theo.tools import MemoryTools

        memory_tools = MemoryTools(
            daemon_client=daemon_client,
            store=sqlite_store,
            validation_loop=validation_loop,
            hybrid_store=hybrid_store,
            settings=settings,
        )
        components["memory_tools"] = memory_tools

        logger.info("Initializing ManagementTools")
        from theo.tools import ManagementTools

        management_tools = ManagementTools(store=sqlite_store)
        components["management_tools"] = management_tools

        logger.info("All components initialized successfully")
        return components

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise


def handle_shutdown(signum: int, _frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown.

    Logs shutdown signal and performs cleanup before exit.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Note: FastMCP handles cleanup automatically
    sys.exit(0)


def call_tool_directly(args: argparse.Namespace) -> None:
    """Call a tool directly and print JSON result to stdout.

    Used by daemon to invoke tools via subprocess without MCP.

    Args:
        args: Parsed CLI arguments with --call and --args
    """
    import asyncio
    import json

    tool_name = args.call
    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON args: {e}"}))
        sys.exit(1)

    # Initialize components
    components = initialize_components(args)

    # Map tool names to methods
    memory_tools = components["memory_tools"]
    query_tools = components["query_tools"]

    tool_map = {
        "memory_store": memory_tools.memory_store,
        "memory_recall": memory_tools.memory_recall,
        "memory_list": memory_tools.memory_list,
        "memory_forget": memory_tools.memory_forget,
        "memory_detect_contradictions": memory_tools.memory_detect_contradictions,
        "memory_relate": memory_tools.memory_relate,
        "memory_apply": memory_tools.memory_apply,
        "memory_outcome": memory_tools.memory_outcome,
        "memory_count": memory_tools.memory_count,
        "memory_analyze_health": memory_tools.memory_analyze_health,
        "doc_search": query_tools.search,
    }

    if tool_name not in tool_map:
        print(json.dumps({"error": f"Unknown tool: {tool_name}"}))
        sys.exit(1)

    method = tool_map[tool_name]

    async def run_tool() -> dict[str, Any]:
        try:
            result = await method(**tool_args)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return {"error": str(e)}

    result = asyncio.run(run_tool())
    print(json.dumps(result))


def trace_query(file: str, line: int | None = None) -> None:
    """Query AI attribution for code."""
    import json

    blame_cmd = ["git", "blame", "--porcelain"]
    if line is not None:
        blame_cmd.extend(["-L", f"{line},{line}"])
    blame_cmd.append(file)

    try:
        result = subprocess.run(blame_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running git blame: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Extract commit SHA (first 40 chars of first line in porcelain format)
    lines = result.stdout.strip().split("\n")
    if not lines:
        print("No blame output", file=sys.stderr)
        sys.exit(1)

    commit_sha = lines[0].split()[0][:40]

    from theo.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    trace = store.get_trace(commit_sha)
    store.close()

    if trace:
        files = json.loads(trace.files_json)
        print(f"AI Attribution Found (spec v{trace.version}):")
        print(f"  Trace ID: {trace.id}")
        print(f"  Timestamp: {trace.timestamp}")
        for f in files:
            print(f"  File: {f['path']}")
            for conv in f.get("conversations", []):
                for r in conv.get("ranges", []):
                    print(f"    Lines {r['start_line']}-{r['end_line']}")
    else:
        print("No AI attribution found for this code")


def main() -> None:
    """Main entry point for MCP server.

    Workflow:
    1. Parse CLI arguments
    2. Setup logging to stderr
    3. Initialize components
    4. Set global tool instances
    5. Register signal handlers
    6. Run MCP server with stdio transport

    Note:
        Uses stdio transport for MCP communication. All logging
        goes to stderr to avoid corrupting JSON-RPC messages on stdout.
    """
    # Parse arguments
    args = parse_arguments()

    # Handle trace subcommands (only exists when trace was the first arg)
    if getattr(args, "subcommand", None) == "trace":
        if args.trace_command == "init":
            trace_init()
            return
        if args.trace_command == "query":
            trace_query(args.file, args.line)
            return
        # No trace subcommand specified, show help
        print("Usage: theo trace {init,query}", file=sys.stderr)
        sys.exit(1)

    # Handle direct tool call mode (for daemon)
    if args.call:
        setup_logging("WARNING")  # Quiet logging for --call mode
        call_tool_directly(args)
        return

    # Setup logging to stderr
    setup_logging(args.log_level)

    logger.info("Starting Theo MCP Server...")
    logger.info(f"Configuration: embedding_backend={args.embedding_backend}")

    try:
        # Initialize all components
        components = initialize_components(args)

        # Set global tool instances in mcp_server module
        from theo.mcp_server import mcp, set_tool_instances

        set_tool_instances(
            indexing=components["indexing_tools"],
            query=components["query_tools"],
            memory=components["memory_tools"],
            management=components["management_tools"],
            store=components["sqlite_store"],
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info("MCP server ready, starting stdio transport...")

        # Run MCP server with stdio transport
        # This blocks until server shuts down
        # Note: mcp.run() is synchronous and manages its own event loop
        mcp.run(transport="stdio")

    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
