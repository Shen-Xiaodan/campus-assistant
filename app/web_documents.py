"""Safe official-site crawling and heading-aware HTML extraction."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from app.documents import _split_oversized, normalize_text
from app.models import TextChunk

USER_AGENT = "CampusKnowledgeAssistant/1.0 (+official-site-indexer)"
SKIP_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".pdf")


@dataclass(frozen=True, slots=True)
class WebSource:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    allowed_paths: tuple[str, ...] = ("/",)
    department: str = ""
    max_pages: int = 50
    max_depth: int = 2
    delay: float = 0.5


@dataclass(slots=True)
class ExtractedWebPage:
    url: str
    title: str
    department: str
    sections: list[tuple[str, str]]
    fields: dict[str, list[str]]
    updated_at: str = ""
    crawled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(
            {"title": self.title, "sections": self.sections, "fields": self.fields},
            ensure_ascii=False,
            sort_keys=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class WebCrawlReport:
    pages: list[ExtractedWebPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CurlResponse:
    content: bytes
    status_code: int
    url: str
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Type": "text/html; charset=utf-8"})

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.url}")


def load_web_sources(path: Path) -> list[WebSource]:
    if not path.exists():
        raise FileNotFoundError(f"官网来源配置不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources: list[WebSource] = []
    for item in payload.get("sources", []):
        start_urls = tuple(str(url).strip() for url in item.get("start_urls", []) if str(url).strip())
        domains = tuple(
            str(domain).lower().strip() for domain in item.get("allowed_domains", []) if str(domain).strip()
        )
        if not item.get("name") or not start_urls or not domains:
            raise ValueError("每个官网来源都必须配置 name、start_urls 和 allowed_domains")
        sources.append(
            WebSource(
                name=str(item["name"]),
                start_urls=start_urls,
                allowed_domains=domains,
                allowed_paths=tuple(item.get("allowed_paths") or ["/"]),
                department=str(item.get("department", "")),
                max_pages=max(1, int(item.get("max_pages", 50))),
                max_depth=max(0, int(item.get("max_depth", 2))),
                delay=max(0.0, float(item.get("delay", 0.5))),
            )
        )
    return sources


def is_allowed_url(url: str, source: WebSource) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    domain_allowed = any(hostname == domain or hostname.endswith(f".{domain}") for domain in source.allowed_domains)
    return (
        parsed.scheme in {"http", "https"}
        and domain_allowed
        and any(parsed.path.startswith(prefix) for prefix in source.allowed_paths)
        and not parsed.path.lower().endswith(SKIP_SUFFIXES)
    )


def extract_structured_fields(text: str) -> dict[str, list[str]]:
    """Extract common contact fields without replacing the original evidence."""

    patterns = {
        "emails": r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])",
        "phones": r"(?:(?:\+?86[-\s]?)?(?:0\d{2,3}[-\s]?)?\d{7,8})(?:[-\s]?转?\d{1,5})?",
        "office_hours": r"(?:办公|工作|服务|开放|接待)(?:时间|时段)[：:]?[^\n。；;]{2,100}",
        "locations": r"(?:办公|联系|服务|办事)(?:地址|地点)[：:]?[^\n。；;]{2,100}",
    }
    result: dict[str, list[str]] = {}
    for key, pattern in patterns.items():
        values = list(dict.fromkeys(match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE)))
        if values:
            result[key] = values[:20]
    return result


def _table_text(table: Tag) -> str:
    rows: list[str] = []
    headers: list[str] = []
    for row_index, row in enumerate(table.find_all("tr")):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if not any(cells):
            continue
        if row_index == 0 and row.find_all("th"):
            headers = cells
            rows.append(" | ".join(cells))
        elif headers and len(headers) == len(cells):
            rows.append("；".join(f"{header}：{value}" for header, value in zip(headers, cells, strict=False)))
        else:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_web_page(html: str, url: str, source: WebSource) -> ExtractedWebPage:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, noscript, nav, footer, header, aside, form, iframe"):
        node.decompose()
    title = normalize_text((soup.title.get_text(" ", strip=True) if soup.title else "") or source.name)
    main = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_parts: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(current_parts))
        if text:
            sections.append((" > ".join(heading_stack) or title, text))
        current_parts.clear()

    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"], recursive=True):
        if node.find_parent(["table", "li"]) and node.name != "table":
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            flush()
            level = int(node.name[1])
            heading = normalize_text(node.get_text(" ", strip=True))
            heading_stack[:] = heading_stack[: level - 1]
            if heading:
                heading_stack.append(heading)
        elif node.name == "table":
            value = _table_text(node)
            if value:
                current_parts.append(value)
        else:
            value = normalize_text(node.get_text(" ", strip=True))
            if value:
                prefix = "- " if node.name == "li" else ""
                current_parts.append(prefix + value)
    flush()
    if not sections:
        fallback = normalize_text(main.get_text("\n", strip=True))
        if fallback:
            sections = [(title, fallback)]
    full_text = "\n".join(text for _, text in sections)
    updated_match = re.search(
        r"(?:更新|发布日期|发布时间|Last\s+Update(?:d)?)[：:\s]*(20\d{2}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)",
        full_text,
        flags=re.IGNORECASE,
    )
    return ExtractedWebPage(
        url=url,
        title=title,
        department=source.department or source.name,
        sections=sections,
        fields=extract_structured_fields(full_text),
        updated_at=updated_match.group(1) if updated_match else "",
    )


def split_web_page(page: ExtractedWebPage, chunk_size: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    field_labels = {"emails": "邮箱", "phones": "电话", "office_hours": "办公时间", "locations": "办公地点"}
    structured = "\n".join(f"{field_labels[key]}：{'；'.join(values)}" for key, values in page.fields.items() if values)
    for section, body in page.sections:
        evidence = f"页面：{page.title}\n章节：{section}\n{body}"
        if structured and any(value in body for values in page.fields.values() for value in values):
            evidence += f"\n\n已识别信息：\n{structured}"
        for piece in _split_oversized(evidence, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    text=piece,
                    metadata={
                        "document": page.title,
                        "source_type": "web",
                        "source_url": page.url,
                        "source_path": page.url,
                        "section": section,
                        "department": page.department,
                        "page": 1,
                        "chunk_index": len(chunks) + 1,
                        "modified_at": page.updated_at,
                        "crawled_at": page.crawled_at,
                        "content_hash": page.content_hash,
                    },
                )
            )
    return chunks


class OfficialSiteCrawler:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robots: dict[str, RobotFileParser | bool] = {}

    def _get(self, url: str) -> requests.Response | CurlResponse:
        try:
            return self.session.get(url, timeout=self.timeout)
        except requests.RequestException as original_error:
            try:
                result = subprocess.run(
                    [
                        "curl",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--max-time",
                        str(self.timeout),
                        "--user-agent",
                        USER_AGENT,
                        "--write-out",
                        "\n%{http_code}",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self.timeout + 5,
                )
                content, status = result.stdout.rsplit(b"\n", 1)
                return CurlResponse(content, int(status), url)
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                raise requests.RequestException(f"requests 与 curl 均无法获取 {url}: {exc}") from original_error

    def _robot_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots:
            try:
                response = self._get(urljoin(root, "/robots.txt"))
                if 200 <= response.status_code < 300:
                    robot = RobotFileParser()
                    robot.set_url(urljoin(root, "/robots.txt"))
                    robot.parse(response.text.splitlines())
                    self._robots[root] = robot
                elif 400 <= response.status_code < 500:
                    # RFC 9309 treats 4xx as "unavailable": crawlers may access
                    # public resources because no applicable rules were served.
                    self._robots[root] = True
                else:
                    self._robots[root] = False
            except requests.RequestException:
                self._robots[root] = False
                return False
        rules = self._robots[root]
        return rules if isinstance(rules, bool) else rules.can_fetch(USER_AGENT, url)

    def crawl(self, source: WebSource) -> WebCrawlReport:
        report = WebCrawlReport()
        queue = [(url, 0) for url in source.start_urls]
        visited: set[str] = set()
        while queue and len(visited) < source.max_pages:
            raw_url, depth = queue.pop(0)
            url = urldefrag(raw_url)[0]
            if url in visited or not is_allowed_url(url, source):
                continue
            visited.add(url)
            if not self._robot_allowed(url):
                report.warnings.append(f"robots.txt 不允许采集: {url}")
                continue
            try:
                response = self._get(url)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                page = extract_web_page(response.text, url, source)
                if sum(len(text) for _, text in page.sections) < 40:
                    report.warnings.append(f"正文过短，未加入索引: {url}")
                    continue
                report.pages.append(page)
                if depth < source.max_depth:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for anchor in soup.find_all("a", href=True):
                        discovered = urldefrag(urljoin(url, anchor["href"]))[0]
                        if discovered not in visited and is_allowed_url(discovered, source):
                            queue.append((discovered, depth + 1))
                if source.delay:
                    time.sleep(source.delay)
            except requests.RequestException as exc:
                report.warnings.append(f"采集失败 {url}: {exc}")
        return report
