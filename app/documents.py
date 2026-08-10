"""PDF parsing, metadata generation and heading-aware text chunking."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from app.exceptions import DocumentProcessingError
from app.models import TextChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParsedPage:
    text: str
    document: str
    source_path: str
    page: int
    modified_at: str | None


@dataclass(slots=True)
class ParseReport:
    pages: list[ParsedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> ParseReport:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise DocumentProcessingError("缺少 pypdf，请先安装项目依赖") from exc

    report = ParseReport()
    try:
        reader = PdfReader(str(path))
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        seen_pages: set[str] = set()
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if not text:
                report.warnings.append(f"{path.name} 第 {page_number} 页无可提取文本，可能是空白页或扫描页")
                continue
            fingerprint = sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen_pages:
                report.warnings.append(f"{path.name} 第 {page_number} 页与前页内容重复，已跳过")
                continue
            seen_pages.add(fingerprint)
            report.pages.append(ParsedPage(text, path.name, str(path.resolve()), page_number, modified))
    except Exception as exc:
        raise DocumentProcessingError(f"解析 {path.name} 失败: {exc}") from exc
    return report


def _split_oversized(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [piece for piece in pieces if piece]


def split_page(page: ParsedPage, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Split by paragraphs first so headings stay attached to following content."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) > chunk_size:
            chunks.extend(_split_oversized(paragraph, chunk_size, overlap))
            current = ""
        else:
            current = paragraph
    if current:
        chunks.append(current)

    result: list[TextChunk] = []
    for index, text in enumerate(chunks, start=1):
        result.append(
            TextChunk(
                text=text,
                metadata={
                    "document": page.document,
                    "source_path": page.source_path,
                    "page": page.page,
                    "chunk_index": index,
                    "modified_at": page.modified_at or "",
                },
            )
        )
    return result


def load_and_split_pdf(path: Path, chunk_size: int, overlap: int) -> tuple[list[TextChunk], list[str]]:
    report = parse_pdf(path)
    chunks = [chunk for page in report.pages for chunk in split_page(page, chunk_size, overlap)]
    if not chunks:
        report.warnings.append(f"{path.name} 未生成任何文本块，扫描版 PDF 请先进行 OCR")
    return chunks, report.warnings
