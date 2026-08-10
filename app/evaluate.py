"""Offline retrieval/citation/refusal evaluation requiring no paid model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.retrieval import keyword_score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(dataset: list[dict[str, Any]], corpus: list[dict[str, Any]], threshold: float = 0.35) -> dict[str, float]:
    retrieval_hits = citation_hits = refusal_hits = 0
    retrieval_cases = citation_cases = refusal_cases = 0
    durations: list[float] = []
    for case in dataset:
        started = time.perf_counter()
        ranked = sorted(
            ((keyword_score(case["question"], item["text"]), item) for item in corpus),
            key=lambda pair: pair[0],
            reverse=True,
        )
        results = [item for score, item in ranked[:3] if score >= threshold]
        refused = not results
        durations.append((time.perf_counter() - started) * 1000)

        refusal_cases += 1
        refusal_hits += int(refused == case["should_refuse"])
        if not case["should_refuse"]:
            retrieval_cases += 1
            hit = any(item["document"] == case["relevant_document"] for item in results)
            retrieval_hits += int(hit)
            citation_cases += 1
            citation_hits += int(
                any(
                    item["document"] == case["relevant_document"] and item["page"] in case["relevant_pages"]
                    for item in results
                )
            )
    return {
        "retrieval_hit_rate": retrieval_hits / retrieval_cases if retrieval_cases else 0.0,
        "citation_accuracy": citation_hits / citation_cases if citation_cases else 0.0,
        "refusal_accuracy": refusal_hits / refusal_cases if refusal_cases else 0.0,
        "average_response_time_ms": sum(durations) / len(durations) if durations else 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="运行无需在线模型的校园问答离线评测")
    parser.add_argument("--dataset", type=Path, default=root / "evaluation" / "dataset.jsonl")
    parser.add_argument("--corpus", type=Path, default=root / "evaluation" / "sample_corpus.json")
    args = parser.parse_args()
    metrics = evaluate(load_jsonl(args.dataset), json.loads(args.corpus.read_text(encoding="utf-8")))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
