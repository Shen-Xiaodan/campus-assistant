"""Download CUHK-Shenzhen major study schemes for a target admission year."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

DEFAULT_INDEX_URL = "https://registry.cuhk.edu.cn/page/22"
USER_AGENT = "College-RAG-study-scheme-downloader/1.0"
ALLOWED_HOST = "registry.cuhk.edu.cn"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class _ContentLinkParser(HTMLParser):
    """Collect links and nearby text from the page's main print-content block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.content_depth = 0
        self.links: list[tuple[str, str]] = []
        self.major_links: list[tuple[str, str]] = []
        self._major_section = False
        self._href: str | None = None
        self._anchor_text: list[str] = []
        self._page_title_depth = 0
        self._page_title: list[str] = []
        self.title = ""
        self._list_depth = 0
        self._list_href: str | None = None
        self._list_text: list[str] = []
        self._list_title_depth = 0
        self._list_title: list[str] = []
        self.documents: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self.content_depth:
            self.content_depth += 1
        elif attributes.get("id") == "print-content":
            self.content_depth = 1
        if not self.content_depth:
            return
        if tag == "h2" and "page-title" in classes:
            self._page_title_depth = 1
        elif self._page_title_depth:
            self._page_title_depth += 1
        if tag == "div" and "list-content" in classes:
            self._list_depth = 1
            self._list_href = None
            self._list_text = []
        elif self._list_depth:
            self._list_depth += 1
        if self._list_depth and tag == "div" and "list-title" in classes:
            self._list_title_depth = 1
            self._list_title = []
        elif self._list_title_depth:
            self._list_title_depth += 1
        if tag == "a" and attributes.get("href"):
            self._href = attributes["href"]
            self._anchor_text = []
            if self._list_depth:
                self._list_href = self._href

    def handle_data(self, data: str) -> None:
        cleaned = _clean_text(data)
        if self.content_depth and cleaned == "主修课程规定":
            self._major_section = True
        elif self.content_depth and cleaned in {"双主修课程", "副修课程规定"}:
            self._major_section = False
        if self._page_title_depth:
            self._page_title.append(data)
        if self._href is not None:
            self._anchor_text.append(data)
        if self._list_depth:
            self._list_text.append(data)
        if self._list_title_depth:
            self._list_title.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.content_depth:
            return
        if tag == "a" and self._href is not None:
            link = (self._href, _clean_text("".join(self._anchor_text)))
            self.links.append(link)
            if self._major_section:
                self.major_links.append(link)
            self._href = None
            self._anchor_text = []
        if self._page_title_depth:
            self._page_title_depth -= 1
            if not self._page_title_depth:
                self.title = _clean_text("".join(self._page_title))
        if self._list_depth:
            self._list_depth -= 1
            if not self._list_depth and self._list_href:
                text = _clean_text("".join(self._list_title or self._list_text))
                match = re.search(r"([^/\\]+\.pdf)\s*$", text, re.IGNORECASE)
                if match:
                    self.documents.append((self._list_href, match.group(1)))
        if self._list_title_depth:
            self._list_title_depth -= 1
        self.content_depth -= 1


def parse_index(html: str, base_url: str) -> list[tuple[str, str]]:
    parser = _ContentLinkParser()
    parser.feed(html)
    seen: set[str] = set()
    majors: list[tuple[str, str]] = []
    for href, label in parser.major_links:
        url = urljoin(base_url, href)
        if label and re.fullmatch(r"https?://registry\.cuhk\.edu\.cn/page/\d+", url) and url not in seen:
            seen.add(url)
            majors.append((label, url))
    return majors


def parse_documents(html: str, base_url: str) -> tuple[str, list[tuple[str, str]]]:
    parser = _ContentLinkParser()
    parser.feed(html)
    return parser.title, [(urljoin(base_url, href), title) for href, title in parser.documents]


def applies_to_year(title: str, admission_year: int) -> bool:
    """Return whether a Chinese study-scheme title covers the admission year."""
    starts = [int(value) for value in re.findall(r"(20\d{2})\s*至\s*\d{2}", title)]
    if admission_year in starts:
        return True
    return bool(starts and ("及以后" in title or "或以后" in title) and admission_year >= starts[-1])


def choose_document(documents: list[tuple[str, str]], admission_year: int) -> tuple[str, str] | None:
    matches = [document for document in documents if applies_to_year(document[1], admission_year)]
    if not matches:
        return None
    exact = [document for document in matches if "及以后" not in document[1] and "或以后" not in document[1]]
    return (exact or matches)[-1]


def safe_filename(value: str) -> str:
    value = unquote(value).replace("/", "_").replace("\\", "_")
    return re.sub(r"[\x00-\x1f:*?\"<>|]", "_", value).strip(" .")


@dataclass(slots=True)
class CrawlRecord:
    major: str
    page_url: str
    status: str
    title: str | None = None
    pdf_url: str | None = None
    path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class HttpContent:
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _get(session: requests.Session, url: str, timeout: float) -> requests.Response | HttpContent:
    if urlparse(url).hostname != ALLOWED_HOST:
        raise ValueError(f"拒绝访问站点范围外的链接: {url}")
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        # This Drupal host currently rejects TLS handshakes from some Python/OpenSSL
        # builds while accepting the system curl client.
        try:
            result = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    str(timeout),
                    "--user-agent",
                    USER_AGENT,
                    url,
                ],
                check=True,
                capture_output=True,
                timeout=timeout + 5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise requests.RequestException(f"requests 与 curl 均无法获取 {url}: {exc}") from exc
        return HttpContent(result.stdout)


def crawl(
    index_url: str,
    output_dir: Path,
    admission_year: int,
    delay: float,
    timeout: float,
    dry_run: bool = False,
) -> list[CrawlRecord]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    index_response = _get(session, index_url, timeout)
    majors = parse_index(index_response.text, index_url)
    if not majors:
        raise RuntimeError("导航页中没有找到专业链接，网站结构可能已经变化")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[CrawlRecord] = []
    for major_label, page_url in majors:
        try:
            if delay:
                time.sleep(delay)
            page_response = _get(session, page_url, timeout)
            page_title, documents = parse_documents(page_response.text, page_url)
            major = page_title or major_label
            selected = choose_document(documents, admission_year)
            if selected is None:
                records.append(CrawlRecord(major, page_url, "not_found"))
                continue
            pdf_url, title = selected
            destination = output_dir / safe_filename(title)
            if dry_run:
                status = "selected"
            elif destination.exists():
                status = "skipped"
            else:
                if delay:
                    time.sleep(delay)
                pdf_response = _get(session, pdf_url, timeout)
                if not pdf_response.content.startswith(b"%PDF"):
                    raise ValueError("下载内容不是有效的 PDF")
                destination.write_bytes(pdf_response.content)
                status = "downloaded"
            records.append(CrawlRecord(major, page_url, status, title, pdf_url, str(destination)))
        except (requests.RequestException, OSError, ValueError) as exc:
            records.append(CrawlRecord(major_label, page_url, "failed", error=str(exc)))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="下载适用于指定入学年份的本科专业修读计划")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL, help="本科生手册导航页")
    parser.add_argument("--year", type=int, default=2023, help="入学年份，默认 2023")
    parser.add_argument("--output", type=Path, default=Path("data/study_schemes/2023"), help="PDF 保存目录")
    parser.add_argument("--delay", type=float, default=0.5, help="请求间隔秒数")
    parser.add_argument("--timeout", type=float, default=30.0, help="单个请求超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只列出选择结果，不下载 PDF")
    args = parser.parse_args()
    if args.year < 2000 or args.delay < 0 or args.timeout <= 0:
        parser.error("year、delay 或 timeout 参数无效")

    records = crawl(args.index_url, args.output, args.year, args.delay, args.timeout, args.dry_run)
    manifest = args.output / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for record in records:
        detail = record.title or record.error or "没有覆盖该入学年份的文件"
        print(f"[{record.status}] {record.major}: {detail}")
    print(f"结果清单: {manifest}")
    return 1 if any(record.status == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
