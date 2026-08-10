"""Backward-compatible wrapper for the new incremental ingestion command."""

from app.ingest import main

if __name__ == "__main__":
    raise SystemExit(main())
