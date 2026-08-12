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
    distance_metric: str = "cosine"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_timeout: float = 120.0
    groq_api_key: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    fetch_k: int = 12
    aggregate_top_k: int = 10
    aggregate_fetch_k: int = 30
    aggregate_max_chunks_per_document: int = 2
    similarity_threshold: float = 0.40
    keyword_bonus_weight: float = 0.25
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
            distance_metric=os.getenv("DISTANCE_METRIC", defaults.distance_metric).strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider).strip().lower(),
            llm_model=os.getenv("LLM_MODEL_ID") or os.getenv("LLM_MODEL", defaults.llm_model),
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or None,
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_timeout=float(os.getenv("LLM_TIMEOUT", str(defaults.llm_timeout))),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            chunk_size=int(os.getenv("CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            top_k=int(os.getenv("TOP_K", str(defaults.top_k))),
            fetch_k=int(os.getenv("FETCH_K", str(defaults.fetch_k))),
            aggregate_top_k=int(os.getenv("AGGREGATE_TOP_K", str(defaults.aggregate_top_k))),
            aggregate_fetch_k=int(os.getenv("AGGREGATE_FETCH_K", str(defaults.aggregate_fetch_k))),
            aggregate_max_chunks_per_document=int(
                os.getenv(
                    "AGGREGATE_MAX_CHUNKS_PER_DOCUMENT",
                    str(defaults.aggregate_max_chunks_per_document),
                )
            ),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", str(defaults.similarity_threshold))),
            keyword_bonus_weight=float(os.getenv("KEYWORD_BONUS_WEIGHT", str(defaults.keyword_bonus_weight))),
            hybrid_search=_env_bool("HYBRID_SEARCH", defaults.hybrid_search),
            max_excerpt_chars=int(os.getenv("MAX_EXCERPT_CHARS", str(defaults.max_excerpt_chars))),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
        )

    def validate(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE 必须大于 0，且 CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.top_k <= 0 or self.fetch_k < self.top_k:
            raise ValueError("TOP_K 必须大于 0，且 FETCH_K 不能小于 TOP_K")
        if self.aggregate_top_k <= 0 or self.aggregate_fetch_k < self.aggregate_top_k:
            raise ValueError("AGGREGATE_TOP_K 必须大于 0，且 AGGREGATE_FETCH_K 不能小于 AGGREGATE_TOP_K")
        if self.aggregate_max_chunks_per_document <= 0:
            raise ValueError("AGGREGATE_MAX_CHUNKS_PER_DOCUMENT 必须大于 0")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("SIMILARITY_THRESHOLD 必须位于 0 到 1 之间")
        if not 0 <= self.keyword_bonus_weight <= 1:
            raise ValueError("KEYWORD_BONUS_WEIGHT 必须位于 0 到 1 之间")
        if self.distance_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("DISTANCE_METRIC 只支持 cosine、l2 或 ip")
        if self.llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT 必须大于 0")
