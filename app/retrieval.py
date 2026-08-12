"""Vector/keyword hybrid retrieval and evidence deduplication."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

from opencc import OpenCC

from app.config import Settings
from app.models import RetrievedEvidence

_T2S = OpenCC("t2s")

# Query variants are intentionally small and domain-oriented. Add new groups as
# real evaluation failures are discovered instead of asking the LLM to rewrite
# every query before retrieval.
QUERY_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "游戏",
        "游戏制作",
        "游戏开发",
        "游戏设计",
        "电子游戏设计与开发",
        "game development",
        "game design",
        "video games design and development",
    ),
    (
        "bio",
        "生物",
        "生命科学",
        "化学与生命科学",
        "biology",
        "biological sciences",
        "life sciences",
    ),
)

# High-precision signals for questions whose answer normally requires collecting
# facts across several documents. Avoid matching a bare ``哪些`` so questions
# such as “申请奖学金需要满足哪些条件” keep the focused retrieval path.
AGGREGATE_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:有|包含|开设|提供)(?:哪|什么)(?:些|几)(?:专业|课程|项目|学院|学系|部门|方向|服务|设施)"),
    re.compile(r"(?:全部|所有|完整)(?:的)?(?:专业|课程|项目|学院|学系|部门|方向|服务|设施|清单|列表)"),
    re.compile(r"(?:列出|罗列|汇总|盘点).{0,12}(?:专业|课程|项目|学院|学系|部门|方向|服务|设施)"),
    re.compile(r"\b(?:list|all)\b.{0,40}\b(?:majors?|programmes?|programs?|courses?|departments?|services?)\b", re.I),
    re.compile(r"\bwhat\b.{0,40}\b(?:majors?|programmes?|programs?|courses?)\b.{0,20}\b(?:offer|available)\w*\b", re.I),
)


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", _T2S.convert(normalized)).strip()


def expand_query(query: str) -> list[str]:
    """Return original query plus Chinese/English variants for matched concepts."""
    normalized_query = normalize_search_text(query)
    variants = [query]
    for group in QUERY_SYNONYM_GROUPS:
        normalized_group = [normalize_search_text(item) for item in group]
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(item)}(?![a-z0-9])", normalized_query) for item in normalized_group
        ):
            variants.extend(group)
    return list(dict.fromkeys(variants))


def is_aggregate_query(query: str) -> bool:
    """Return whether a question asks for a multi-item catalogue or summary."""
    normalized = normalize_search_text(query)
    return any(pattern.search(normalized) for pattern in AGGREGATE_QUERY_PATTERNS)


def _tokens(text: str) -> list[str]:
    lowered = normalize_search_text(text)
    latin = re.findall(r"[a-z0-9]{2,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = ["".join(chinese[index : index + 2]) for index in range(max(0, len(chinese) - 1))]
    return latin + chinese + bigrams


def keyword_score(query: str, text: str) -> float:
    query_counts = Counter(_tokens(query))
    document_counts = Counter(_tokens(text))
    if not query_counts or not document_counts:
        return 0.0
    matched = sum(min(count, document_counts[token]) for token, count in query_counts.items())
    return min(1.0, matched / sum(query_counts.values()))


def deduplicate_evidence(items: Iterable[RetrievedEvidence], top_k: int) -> list[RetrievedEvidence]:
    best: dict[tuple[str, int | None, str], RetrievedEvidence] = {}
    for item in items:
        normalized = re.sub(r"\s+", " ", item.text).strip().lower()
        key = (item.document, item.page, normalized)
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda item: item.score, reverse=True)[:top_k]


def diversify_evidence(
    items: Iterable[RetrievedEvidence],
    top_k: int,
    max_chunks_per_document: int,
) -> list[RetrievedEvidence]:
    """Select strong evidence in rounds so one document cannot fill every slot."""
    deduplicated = deduplicate_evidence(items, top_k=10**9)
    groups: dict[str, list[RetrievedEvidence]] = {}
    document_order: list[str] = []
    for item in deduplicated:
        identity = str(item.metadata.get("source_url") or item.document)
        if identity not in groups:
            groups[identity] = []
            document_order.append(identity)
        groups[identity].append(item)

    selected: list[RetrievedEvidence] = []
    for chunk_index in range(max_chunks_per_document):
        round_items = [
            groups[identity][chunk_index]
            for identity in document_order
            if len(groups[identity]) > chunk_index
        ]
        round_items.sort(key=lambda item: item.score, reverse=True)
        selected.extend(round_items[: max(0, top_k - len(selected))])
        if len(selected) >= top_k:
            break
    return selected


def distance_to_relevance(distance: float, metric: str) -> float:
    """Convert Chroma's lower-is-better raw distance to a bounded relevance score."""
    distance = max(0.0, float(distance))
    if metric == "cosine":
        score = 1.0 - distance
    elif metric == "l2":
        score = 1.0 / (1.0 + distance)
    elif metric == "ip":
        score = 1.0 - distance
    else:
        raise ValueError(f"不支持的距离类型: {metric}")
    return max(0.0, min(1.0, score))


class CampusRetriever:
    def __init__(self, vectorstore: Any, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings

    @staticmethod
    def _evidence(document: Any, score: float) -> RetrievedEvidence:
        metadata = dict(getattr(document, "metadata", {}) or {})
        source_type = str(metadata.get("source_type", "pdf"))
        return RetrievedEvidence(
            text=getattr(document, "page_content", ""),
            document=str(metadata.get("document", "未知文档")),
            page=None if source_type == "web" else max(1, int(metadata.get("page", 1))),
            score=max(0.0, min(1.0, float(score))),
            metadata=metadata,
        )

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        aggregate_query = is_aggregate_query(query)
        fetch_k = self.settings.aggregate_fetch_k if aggregate_query else self.settings.fetch_k
        top_k = self.settings.aggregate_top_k if aggregate_query else self.settings.top_k
        query_variants = expand_query(query)
        vector_results: dict[tuple[str, int, str], tuple[Any, float]] = {}
        for variant in query_variants:
            for document, distance in self.vectorstore.similarity_search_with_score(variant, k=fetch_k):
                metadata = document.metadata or {}
                key = (
                    str(metadata.get("source_url") or metadata.get("document", "")),
                    int(metadata.get("page", 1)),
                    document.page_content,
                )
                relevance = distance_to_relevance(distance, self.settings.distance_metric)
                previous = vector_results.get(key)
                if previous is None or relevance > previous[1]:
                    vector_results[key] = (document, relevance)

        merged: list[RetrievedEvidence] = []
        seen_texts: set[str] = set()
        for document, vector_score in vector_results.values():
            lexical = (
                max(keyword_score(variant, document.page_content) for variant in query_variants)
                if self.settings.hybrid_search
                else 0.0
            )
            combined = min(1.0, vector_score + self.settings.keyword_bonus_weight * lexical)
            evidence = self._evidence(document, combined)
            merged.append(evidence)
            seen_texts.add(document.page_content)

        if self.settings.hybrid_search:
            raw = self.vectorstore.get(include=["documents", "metadatas"])
            keyword_candidates: list[RetrievedEvidence] = []
            for text, metadata in zip(raw.get("documents", []), raw.get("metadatas", []), strict=False):
                score = max(keyword_score(variant, text) for variant in query_variants)
                if score > 0 and text not in seen_texts:
                    holder = type("Document", (), {"page_content": text, "metadata": metadata})()
                    keyword_candidates.append(self._evidence(holder, score))
            merged.extend(
                sorted(keyword_candidates, key=lambda item: item.score, reverse=True)[:fetch_k]
            )

        eligible = [item for item in merged if item.score >= self.settings.similarity_threshold]
        if aggregate_query:
            return diversify_evidence(
                eligible,
                top_k=top_k,
                max_chunks_per_document=self.settings.aggregate_max_chunks_per_document,
            )
        return deduplicate_evidence(eligible, top_k)
