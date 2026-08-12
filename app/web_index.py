"""Incremental vector indexing for official-site pages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from app.config import Settings
from app.index import MANIFEST_NAME, _load_manifest, open_vectorstore
from app.web_documents import OfficialSiteCrawler, load_web_sources, split_web_page


@dataclass(slots=True)
class WebIndexReport:
    crawled_pages: int = 0
    indexed_pages: int = 0
    skipped_pages: int = 0
    chunks_added: int = 0
    warnings: list[str] = field(default_factory=list)


def _web_chunk_id(url: str, content_hash: str, index: int) -> str:
    return sha256(f"web:{url}:{content_hash}:{index}".encode()).hexdigest()


def build_web_index(settings: Settings, config_path: Path, timeout: float = 20.0) -> WebIndexReport:
    report = WebIndexReport()
    sources = load_web_sources(config_path)
    manifest_path = settings.index_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    web_manifest = manifest.setdefault("web_documents", {})
    vectorstore = None
    crawler = OfficialSiteCrawler(timeout=timeout)

    for source in sources:
        crawl_report = crawler.crawl(source)
        report.warnings.extend(crawl_report.warnings)
        report.crawled_pages += len(crawl_report.pages)
        for page in crawl_report.pages:
            previous = web_manifest.get(page.url)
            if previous and previous.get("content_hash") == page.content_hash:
                report.skipped_pages += 1
                continue
            chunks = split_web_page(page, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                report.warnings.append(f"网页未生成文本块: {page.url}")
                continue
            if vectorstore is None:
                vectorstore = open_vectorstore(settings)
            if previous and previous.get("chunk_ids"):
                vectorstore.delete(ids=previous["chunk_ids"])
            ids = [_web_chunk_id(page.url, page.content_hash, index) for index in range(1, len(chunks) + 1)]
            vectorstore.add_texts(
                texts=[chunk.text for chunk in chunks], metadatas=[chunk.metadata for chunk in chunks], ids=ids
            )
            web_manifest[page.url] = {
                "content_hash": page.content_hash,
                "chunk_ids": ids,
                "document": page.title,
                "crawled_at": page.crawled_at,
            }
            report.indexed_pages += 1
            report.chunks_added += len(chunks)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
