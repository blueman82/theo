"""Integration tests for Theo.

This package contains comprehensive integration tests that verify
all components work together correctly:

- test_full_pipeline.py: Full indexing pipeline (file -> chunks -> embeddings -> storage)
- test_daemon_integration.py: Daemon IPC (client -> socket -> server -> worker)
- test_mcp_tools.py: MCP tool handlers (request -> handler -> response)

Integration tests use:
- Temporary directories for file operations
- Temporary ChromaDB instances for data isolation
- Mock embedding providers to avoid external dependencies
- Real component interactions (not mocked)

Usage:
    # Run all integration tests
    uv run pytest tests/integration/ -v

    # Run specific test file
    uv run pytest tests/integration/test_full_pipeline.py -v

    # Run with coverage
    uv run pytest tests/integration/ -v --cov=src/theo
"""
