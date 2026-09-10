"""Vector/keyword hybrid retrieval and evidence deduplication."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from math import log
from typing import Any, Iterable, Protocol, Sequence

from opencc import OpenCC

from app.config import Settings
from app.models import RetrievedEvidence

_T2S = OpenCC("t2s")

# Query variants are intentionally small and domain-oriented. Add new groups as
# real evaluation failures are discovered instead of asking the LLM to rewrite
# every query before retrieval.
QUERY_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("学生事务处", "学生事务", "student affairs", "office of student affairs", "osa"),
    ("教务处", "教务", "registry", "academic affairs"),
    ("图书馆", "library"),
    ("奖学金", "scholarship", "financial aid"),
    ("学生证", "校园卡", "student card", "campus card"),
    ("培养方案", "修读计划", "programme requirements", "program requirements", "study scheme"),
    ("必修课", "必修课程", "required course", "required courses", "compulsory course"),
    ("选修课", "选修课程", "elective", "elective course", "elective courses"),
    ("学分", "credit", "credits"),
    ("本科生", "本科", "undergraduate", "undergraduates"),
    ("研究生", "硕士生", "博士生", "postgraduate", "graduate student", "graduate students"),
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
    # Match course codes despite PDF extraction inserting spaces, e.g.
    # ``MA T1001`` vs the user query ``MAT1001``.
    normalized = re.sub(r"\b([a-z]{2,4})\s*t\s*(\d{4}[a-z]?)\b", r"\1t\2", normalized)
    normalized = re.sub(r"\b([a-z]{2,4})\s+(\d{4}[a-z]?)\b", r"\1\2", normalized)
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


def matching_study_scheme_documents(query: str, metadatas: Iterable[dict[str, Any] | None]) -> set[str]:
    """Find explicitly named programme documents, preferring the longest name."""
    normalized_query = normalize_search_text(query)
    matches: dict[str, str] = {}
    for metadata in metadatas:
        document = str((metadata or {}).get("document", ""))
        if not document.lower().endswith(".pdf") or "_适用于" not in document:
            continue
        programme = document.split("_适用于", 1)[0].strip()
        if programme and normalize_search_text(programme) in normalized_query:
            matches[document] = programme
    if not matches:
        return set()
    longest = max(len(normalize_search_text(programme)) for programme in matches.values())
    return {
        document
        for document, programme in matches.items()
        if len(normalize_search_text(programme)) == longest
    }


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


class BM25Index:
    """Small in-memory BM25 index built once from the persisted Chroma corpus."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(document) for document in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequencies: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequencies.update(set(tokens))
        count = len(self.documents)
        self.idf = {
            token: log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }

    def search(self, query_variants: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        query_tokens = list(dict.fromkeys(token for variant in query_variants for token in _tokens(variant)))
        scores: list[tuple[int, float]] = []
        for index, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            length_normalization = 1 - self.b
            if self.average_length:
                length_normalization += self.b * self.lengths[index] / self.average_length
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_normalization
                score += self.idf.get(token, 0.0) * numerator / denominator
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


class Reranker(Protocol):
    def rank(self, query: str, evidence: Sequence[RetrievedEvidence]) -> list[float]: ...


class BGEReranker:
    """Lazy FlagEmbedding adapter; only imported when reranking is enabled."""

    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError("启用 reranker 需要安装 FlagEmbedding") from exc
        self.model = FlagReranker(model_name, use_fp16=False)

    def rank(self, query: str, evidence: Sequence[RetrievedEvidence]) -> list[float]:
        scores = self.model.compute_score([[query, item.text] for item in evidence], normalize=True)
        return [float(scores)] if isinstance(scores, (float, int)) else [float(score) for score in scores]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedEvidence]],
    rrf_k: int = 60,
) -> list[RetrievedEvidence]:
    scores: defaultdict[tuple[str, int | None, str], float] = defaultdict(float)
    evidence_by_key: dict[tuple[str, int | None, str], RetrievedEvidence] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = (item.document, item.page, item.text)
            scores[key] += 1 / (rrf_k + rank)
            evidence_by_key[key] = item
    if not scores:
        return []
    max_score = max(scores.values())
    fused = [
        RetrievedEvidence(
            text=evidence_by_key[key].text,
            document=evidence_by_key[key].document,
            page=evidence_by_key[key].page,
            score=score / max_score,
            metadata=evidence_by_key[key].metadata,
        )
        for key, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)


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
    def __init__(self, vectorstore: Any, settings: Settings, reranker: Reranker | None = None):
        self.vectorstore = vectorstore
        self.settings = settings
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        self.documents = list(raw.get("documents", []))
        self.metadatas = list(raw.get("metadatas", []))
        self.bm25 = BM25Index(self.documents)
        self.reranker = reranker
        catalog_path = settings.data_dir / "course_catalog.json"
        try:
            self.course_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.course_catalog = []

    def _catalog_lookup(self, query: str) -> list[RetrievedEvidence]:
        """Return exact course-code hits before approximate vector ranking."""
        normalized = normalize_search_text(query).upper()
        # ``\\b`` is unreliable next to Chinese characters because Python
        # treats them as word characters; use ASCII-aware lookarounds.
        codes = set(re.findall(r"(?<![A-Z0-9])[A-Z]{2,4}\d{4}[A-Z]?(?![A-Z0-9])", normalized))
        if not codes:
            return []
        results: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str]] = set()
        for record in self.course_catalog:
            if record.get("course_code") not in codes:
                continue
            excerpt = str(record.get("excerpt", "")).strip()
            # Ignore code-only requirement lists when a titled course row is
            # available elsewhere in the catalogue.
            if (
                len(re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", excerpt)) <= len(record["course_code"])
                or not re.search(r"\s\d+(?:\.\d+)?$", excerpt)
            ):
                continue
            key = (record["course_code"], record.get("source_document", ""), excerpt)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                RetrievedEvidence(
                    text=excerpt,
                    document=record["source_document"],
                    page=int(record["page"]),
                    score=1.0,
                    metadata={"source_type": "course_catalog", "page": int(record["page"])},
                )
            )
        return results[: self.settings.top_k]

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
        catalog_hits = self._catalog_lookup(query)
        if catalog_hits:
            return catalog_hits
        aggregate_query = is_aggregate_query(query)
        fetch_k = self.settings.aggregate_fetch_k if aggregate_query else self.settings.fetch_k
        top_k = self.settings.aggregate_top_k if aggregate_query else self.settings.top_k
        query_variants = expand_query(query)
        target_documents = matching_study_scheme_documents(query, self.metadatas)
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

        vector_ranking: list[RetrievedEvidence] = []
        for document, vector_score in vector_results.values():
            vector_ranking.append(self._evidence(document, vector_score))
        vector_ranking.sort(key=lambda item: item.score, reverse=True)

        rankings = [vector_ranking[:fetch_k]]
        if self.settings.hybrid_search:
            bm25_ranking = []
            for index, _score in self.bm25.search(query_variants, fetch_k):
                lexical_relevance = max(
                    keyword_score(variant, self.documents[index]) for variant in query_variants
                )
                if lexical_relevance < self.settings.similarity_threshold:
                    continue
                holder = type(
                    "Document",
                    (),
                    {"page_content": self.documents[index], "metadata": self.metadatas[index]},
                )()
                bm25_ranking.append(self._evidence(holder, lexical_relevance))
            rankings.append(bm25_ranking)

        eligible = reciprocal_rank_fusion(rankings, self.settings.rrf_k)
        # A strong match in either candidate list remains eligible; the RRF
        # score is for ordering and is normalized independently per query.
        candidate_keys = {
            (item.document, item.page, item.text)
            for item in vector_ranking
            if item.score >= self.settings.similarity_threshold
        }
        if len(rankings) > 1:
            candidate_keys.update((item.document, item.page, item.text) for item in rankings[1])
        eligible = [item for item in eligible if (item.document, item.page, item.text) in candidate_keys]
        if target_documents:
            eligible = [item for item in eligible if item.document in target_documents]

        if self.settings.reranker_enabled:
            reranker = self.reranker or BGEReranker(self.settings.reranker_model)
            candidates = eligible[: self.settings.reranker_candidates]
            reranker_scores = reranker.rank(query, candidates)
            eligible = [
                RetrievedEvidence(item.text, item.document, item.page, score, item.metadata)
                for item, score in zip(candidates, reranker_scores, strict=True)
            ]
            eligible.sort(key=lambda item: item.score, reverse=True)
        if aggregate_query:
            return diversify_evidence(
                eligible,
                top_k=top_k,
                max_chunks_per_document=self.settings.aggregate_max_chunks_per_document,
            )
        return deduplicate_evidence(eligible, top_k)
