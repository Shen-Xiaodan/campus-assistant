"""Turn tool evidence into a cited chat response."""

from __future__ import annotations

from app.generation import REFUSAL_ANSWER, AnswerGenerator, citation_label
from app.models import ChatHistoryMessage, ChatResponse, RetrievedEvidence, SourceResponse
from app.tool_models import AgentRunResult


def evidence_from_run(run: AgentRunResult) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    seen: set[tuple[str, int | None, str]] = set()
    for step in run.steps:
        if not step.result.success:
            continue
        for item in step.result.evidence:
            key = (item.document, item.page, item.text)
            if key in seen:
                continue
            seen.add(key)
            metadata = dict(item.metadata)
            metadata.update(source_type=item.source_type)
            if item.url:
                metadata["source_url"] = item.url
            if item.section:
                metadata["section"] = item.section
            evidence.append(RetrievedEvidence(item.text, item.document, item.page, item.score, metadata))
    return evidence


def answer_from_run(
    run: AgentRunResult,
    question: str,
    history: list[ChatHistoryMessage] | None,
    generator: AnswerGenerator,
) -> ChatResponse | None:
    if not run.steps or (not run.completed and run.stop_reason not in {"planner_error", "max_steps"}):
        return None
    if any(not step.result.success for step in run.steps):
        return None
    evidence = evidence_from_run(run)
    for step in run.steps:
        if step.tool_call.name != "compare_scheme_versions" or not isinstance(step.result.data, dict):
            continue
        versions = step.result.data.get("versions", [])
        if not isinstance(versions, list):
            continue
        missing = [str(version["admission_year"]) for version in versions if not version.get("evidence_indexes")]
        if missing:
            sources = [
                SourceResponse(
                    document=item.document,
                    page=item.page,
                    excerpt=item.text[: generator.settings.max_excerpt_chars],
                    score=item.score,
                    source_type=str(item.metadata.get("source_type", "pdf")),
                )
                for item in evidence[:3]
            ]
            citation = f" {citation_label(evidence[0].document, evidence[0].page)}" if evidence else ""
            return ChatResponse(
                answer=(
                    f"目前只找到部分版本的培养方案资料{citation}；"
                    f"未找到 {', '.join(missing)} 年对应版本，无法可靠比较差异。"
                ),
                sources=sources,
                grounded=False,
                answer_mode="official_fact",
            )
    if not evidence:
        return None
    response = generator.answer(question, evidence, history)
    return response if response.grounded else ChatResponse(
        answer=REFUSAL_ANSWER,
        sources=[],
        grounded=False,
        answer_mode="official_fact",
    )
