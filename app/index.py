"""Persistent Chroma index construction with hash-based incremental updates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.config import Settings
from app.documents import file_sha256, load_and_split_pdf
from app.exceptions import IndexUnavailableError

logger = logging.getLogger(__name__)
MANIFEST_NAME = "index_manifest.json"


@dataclass(slots=True)
class IndexReport:
    indexed_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    chunks_added: int = 0
    warnings: list[str] = field(default_factory=list)


def create_embeddings(settings: Settings):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise IndexUnavailableError("缺少 langchain-huggingface，请安装项目依赖") from exc
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def open_vectorstore(settings: Settings, embeddings: Any | None = None):
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise IndexUnavailableError("缺少 langchain-chroma，请安装项目依赖") from exc
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.index_dir),
        embedding_function=embeddings or create_embeddings(settings),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "documents": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("index_manifest_invalid", extra={"path": str(path)})
        return {"version": 1, "documents": {}}


def _chunk_id(file_hash: str, page: int, chunk_index: int, text: str) -> str:
    value = f"{file_hash}:{page}:{chunk_index}:{text}".encode("utf-8")
    return sha256(value).hexdigest()


def build_index(settings: Settings, source_dir: Path | None = None) -> IndexReport:
    source_dir = source_dir or settings.data_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.index_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    documents_manifest: dict[str, Any] = manifest.setdefault("documents", {})
    report = IndexReport()
    vectorstore = None

    pdf_paths = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdf_paths:
        report.warnings.append(f"{source_dir} 中没有 PDF 文档")

    for path in pdf_paths:
        key = str(path.resolve())
        try:
            current_hash = file_sha256(path)
            old_entry = documents_manifest.get(key)
            if old_entry and old_entry.get("sha256") == current_hash:
                report.skipped_documents += 1
                continue
            chunks, warnings = load_and_split_pdf(path, settings.chunk_size, settings.chunk_overlap)
            report.warnings.extend(warnings)
            if not chunks:
                report.failed_documents += 1
                continue
            if vectorstore is None:
                vectorstore = open_vectorstore(settings)
            if old_entry and old_entry.get("chunk_ids"):
                vectorstore.delete(ids=old_entry["chunk_ids"])
            ids = [
                _chunk_id(current_hash, int(chunk.metadata["page"]), int(chunk.metadata["chunk_index"]), chunk.text)
                for chunk in chunks
            ]
            texts = [chunk.text for chunk in chunks]
            metadatas = [{**chunk.metadata, "file_hash": current_hash} for chunk in chunks]
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            documents_manifest[key] = {"sha256": current_hash, "chunk_ids": ids, "document": path.name}
            report.indexed_documents += 1
            report.chunks_added += len(chunks)
        except Exception as exc:
            report.failed_documents += 1
            message = f"{path.name} 索引失败: {exc}"
            report.warnings.append(message)
            logger.exception("document_index_failed")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
