"""Vector/keyword hybrid retrieval and evidence deduplication."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from app.config import Settings
from app.models import RetrievedEvidence


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
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
    best: dict[tuple[str, int, str], RetrievedEvidence] = {}
    for item in items:
        normalized = re.sub(r"\s+", " ", item.text).strip().lower()
        key = (item.document, item.page, normalized)
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda item: item.score, reverse=True)[:top_k]


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
        return RetrievedEvidence(
            text=getattr(document, "page_content", ""),
            document=str(metadata.get("document", "未知文档")),
            page=max(1, int(metadata.get("page", 1))),
            score=max(0.0, min(1.0, float(score))),
            metadata=metadata,
        )

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        vector_results = self.vectorstore.similarity_search_with_score(query, k=self.settings.fetch_k)
        merged: list[RetrievedEvidence] = []
        seen_texts: set[str] = set()
        for document, distance in vector_results:
            vector_score = distance_to_relevance(distance, self.settings.distance_metric)
            lexical = keyword_score(query, document.page_content) if self.settings.hybrid_search else 0.0
            combined = (
                0.75 * float(vector_score) + 0.25 * lexical if self.settings.hybrid_search else float(vector_score)
            )
            evidence = self._evidence(document, combined)
            merged.append(evidence)
            seen_texts.add(document.page_content)

        if self.settings.hybrid_search:
            raw = self.vectorstore.get(include=["documents", "metadatas"])
            keyword_candidates: list[RetrievedEvidence] = []
            for text, metadata in zip(raw.get("documents", []), raw.get("metadatas", []), strict=False):
                score = keyword_score(query, text)
                if score > 0 and text not in seen_texts:
                    holder = type("Document", (), {"page_content": text, "metadata": metadata})()
                    keyword_candidates.append(self._evidence(holder, score * 0.65))
            merged.extend(
                sorted(keyword_candidates, key=lambda item: item.score, reverse=True)[: self.settings.fetch_k]
            )

        eligible = [item for item in merged if item.score >= self.settings.similarity_threshold]
        return deduplicate_evidence(eligible, self.settings.top_k)
