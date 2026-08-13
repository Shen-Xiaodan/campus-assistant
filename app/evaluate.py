"""Offline evaluation for the production BM25/RRF retrieval pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config import Settings
from app.retrieval import CampusRetriever, keyword_score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EvaluationVectorstore:
    """Deterministic vectorstore stand-in that exercises the real hybrid pipeline."""

    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus

    def get(self, include):
        return {
            "documents": [item["text"] for item in self.corpus],
            "metadatas": [
                {"document": item["document"], "page": item["page"], **item.get("metadata", {})}
                for item in self.corpus
            ],
        }

    def similarity_search_with_score(self, query: str, k: int):
        ranked = sorted(
            self.corpus,
            key=lambda item: keyword_score(query, item["text"]),
            reverse=True,
        )[:k]
        return [
            (
                SimpleNamespace(
                    page_content=item["text"],
                    metadata={"document": item["document"], "page": item["page"], **item.get("metadata", {})},
                ),
                1 - keyword_score(query, item["text"]),
            )
            for item in ranked
        ]


def evaluate(
    dataset: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    threshold: float = 0.35,
) -> dict[str, float]:
    settings = Settings(
        top_k=4,
        fetch_k=20,
        aggregate_top_k=10,
        aggregate_fetch_k=20,
        similarity_threshold=threshold,
    )
    retriever = CampusRetriever(EvaluationVectorstore(corpus), settings)
    recall_hits = reciprocal_ranks = page_hits = refusal_hits = 0.0
    answerable_cases = refusal_cases = 0
    durations: list[float] = []

    for case in dataset:
        started = time.perf_counter()
        results = retriever.retrieve(case["question"])
        durations.append((time.perf_counter() - started) * 1000)
        should_refuse = case["should_refuse"]
        refusal_cases += 1
        refusal_hits += float((not results) == should_refuse)
        if should_refuse:
            continue

        answerable_cases += 1
        expected_document = case["relevant_document"]
        matching_ranks = [index for index, item in enumerate(results, start=1) if item.document == expected_document]
        if matching_ranks:
            recall_hits += 1
            reciprocal_ranks += 1 / matching_ranks[0]
        page_hits += float(
            any(item.document == expected_document and item.page in case["relevant_pages"] for item in results)
        )

    recall = recall_hits / answerable_cases if answerable_cases else 0.0
    page_hit_rate = page_hits / answerable_cases if answerable_cases else 0.0
    return {
        "recall_at_5": recall,
        "mrr_at_10": reciprocal_ranks / answerable_cases if answerable_cases else 0.0,
        "page_hit_rate": page_hit_rate,
        "refusal_accuracy": refusal_hits / refusal_cases if refusal_cases else 0.0,
        "average_response_time_ms": sum(durations) / len(durations) if durations else 0.0,
        # Backward-compatible names retained for existing dashboards.
        "retrieval_hit_rate": recall,
        "citation_accuracy": page_hit_rate,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="运行 BM25/RRF 校园检索离线评测")
    parser.add_argument("--dataset", type=Path, default=root / "evaluation" / "dataset.jsonl")
    parser.add_argument("--corpus", type=Path, default=root / "evaluation" / "sample_corpus.json")
    args = parser.parse_args()
    metrics = evaluate(load_jsonl(args.dataset), json.loads(args.corpus.read_text(encoding="utf-8")))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
