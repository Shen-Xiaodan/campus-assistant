"""Application service composing retrieval and generation."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from app.generation import AnswerGenerator
from app.language import detect_language
from app.models import ChatHistoryMessage, ChatResponse, RetrievedEvidence


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...


_APPLICABILITY_PATTERN = re.compile(r"^(?P<programme>.+)_适用于(?P<label>.+?)\.pdf$", re.I)
_YEAR_PATTERN = re.compile(r"(?:20)?\d{2}\s*(?:至|[-–—/])\s*(?:20)?\d{2}|20\d{2}\s*(?:级|年|entry)", re.I)


def clarification_for_versions(question: str, evidence: list[RetrievedEvidence]) -> tuple[str, list[str]] | None:
    """Ask only when retrieved evidence contains competing versions of one programme."""
    if _YEAR_PATTERN.search(question):
        return None
    versions: defaultdict[str, set[str]] = defaultdict(set)
    for item in evidence:
        match = _APPLICABILITY_PATTERN.match(item.document)
        if match:
            versions[match.group("programme")].add(match.group("label"))
    ambiguous = [(programme, labels) for programme, labels in versions.items() if len(labels) > 1]
    if not ambiguous:
        return None
    programme, labels = max(ambiguous, key=lambda item: len(item[1]))
    options = [f"适用于{label}" for label in sorted(labels)]
    if detect_language(question) == "en":
        message = (
            f"I found more than one version of the {programme} study scheme. "
            "Which entry-year version would you like me to use?"
        )
    else:
        message = f"我找到了多个版本的{programme}培养方案。为了不把不同年份的要求混在一起，你想看哪一版？"
    return message, options


class QAService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, history: list[ChatHistoryMessage] | None = None) -> ChatResponse:
        clean_question = question.strip()
        recent_history = (history or [])[-6:]
        previous_user_questions = [item.content for item in recent_history if item.role == "user"]
        retrieval_query = "\n相关上文：".join([*previous_user_questions[-2:], clean_question])
        evidence = self.retriever.retrieve(retrieval_query)
        clarification = clarification_for_versions(clean_question, evidence)
        if clarification:
            message, options = clarification
            return ChatResponse(
                answer=message,
                sources=[],
                grounded=False,
                needs_clarification=True,
                clarification_options=options,
            )
        return self.generator.answer(clean_question, evidence, recent_history)
