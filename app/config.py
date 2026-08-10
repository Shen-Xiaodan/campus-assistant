"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "校园知识问答助手"
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "vector_db_dir"
    collection_name: str = "campus_documents"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    fetch_k: int = 12
    similarity_threshold: float = 0.45
    hybrid_search: bool = True
    max_excerpt_chars: int = 240
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        defaults = cls()
        return cls(
            app_name=os.getenv("APP_NAME", defaults.app_name),
            data_dir=Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve(),
            index_dir=Path(os.getenv("INDEX_DIR", str(PROJECT_ROOT / "vector_db_dir"))).expanduser().resolve(),
            collection_name=os.getenv("CHROMA_COLLECTION", defaults.collection_name),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            chunk_size=int(os.getenv("CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            top_k=int(os.getenv("TOP_K", str(defaults.top_k))),
            fetch_k=int(os.getenv("FETCH_K", str(defaults.fetch_k))),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", str(defaults.similarity_threshold))),
            hybrid_search=_env_bool("HYBRID_SEARCH", defaults.hybrid_search),
            max_excerpt_chars=int(os.getenv("MAX_EXCERPT_CHARS", str(defaults.max_excerpt_chars))),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
        )

    def validate(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE 必须大于 0，且 CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.top_k <= 0 or self.fetch_k < self.top_k:
            raise ValueError("TOP_K 必须大于 0，且 FETCH_K 不能小于 TOP_K")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("SIMILARITY_THRESHOLD 必须位于 0 到 1 之间")
