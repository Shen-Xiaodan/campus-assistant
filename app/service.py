"""Application service composing retrieval and generation."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from app.generation import OFFICIAL_FACT, AnswerGenerator, classify_query
from app.language import detect_language
from app.models import ChatHistoryMessage, ChatResponse, RetrievedEvidence


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...


_APPLICABILITY_PATTERN = re.compile(r"^(?P<programme>.+)_适用于(?P<label>.+?)\.pdf$", re.I)
_YEAR_PATTERN = re.compile(r"(?:20)?\d{2}\s*(?:至|[-–—/])\s*(?:20)?\d{2}|20\d{2}\s*(?:级|年|entry)", re.I)


def applicability_option(label: str, language: str) -> str:
    if language == "zh":
        return f"适用于{label}"
    english = (
        label.replace("至", "–")
        .replace("、", ", ")
        .replace("年度及以后入学学生", " academic year and later entrants")
        .replace("年度入学学生", " academic year entrants")
        .replace("及", " and ")
    )
    return f"For {english}"


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
    language = detect_language(question)
    options = [applicability_option(label, language) for label in sorted(labels)]
    if language == "en":
        message = (
            f"I found more than one version of the {programme} study scheme. "
            "Which entry-year version would you like me to use?"
        )
    else:
        message = f"我找到了多个版本的{programme}培养方案。为了不把不同年份的要求混在一起，你想看哪一版？"
    return message, options


def answer_scope(
    question: str,
    evidence: list[RetrievedEvidence],
) -> tuple[list[RetrievedEvidence], str, list[str]] | None:
    """Select one explicit study-scheme version and describe the answer boundary."""
    if _YEAR_PATTERN.search(question):
        return None
    scoped_items: list[tuple[RetrievedEvidence, str, str]] = []
    for item in evidence:
        match = _APPLICABILITY_PATTERN.match(item.document)
        if match:
            scoped_items.append((item, match.group("programme"), match.group("label")))
    if not scoped_items:
        return None

    selected_item, selected_programme, selected_label = scoped_items[0]
    selected_document = selected_item.document
    scoped_evidence = [
        item
        for item in evidence
        if not _APPLICABILITY_PATTERN.match(item.document) or item.document == selected_document
    ]
    other_labels = sorted(
        {
            label
            for _, programme, label in scoped_items
            if programme == selected_programme and label != selected_label
        }
    )
    language = detect_language(question)
    options = [applicability_option(label, language) for label in other_labels]
    if language == "en":
        selected_scope = applicability_option(selected_label, language).removeprefix("For ")
        notice = f"This answer uses the {selected_programme} study scheme for {selected_scope}."
        if options:
            notice += " Requirements may differ for other entry years; you can switch versions below."
    else:
        notice = f"目前按适用于{selected_label}的{selected_programme}培养方案回答。"
        if options:
            notice += "其他入学年份的要求可能不同，你也可以切换版本继续查看。"
    return scoped_evidence, notice, options


class QAService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def ask(
        self,
        question: str,
        history: list[ChatHistoryMessage] | None = None,
        connection: ModelConnection | None = None,
    ) -> ChatResponse:
        clean_question = question.strip()
        recent_history = (history or [])[-6:]
        previous_user_questions = [item.content for item in recent_history if item.role == "user"]
        retrieval_query = "\n相关上文：".join([*previous_user_questions[-2:], clean_question])
        evidence = self.retriever.retrieve(retrieval_query)
        retrieved_evidence = evidence
        # A course explanation/plan is not tied to the first programme that
        # happens to mention a shared course. Version scoping is required only
        # for official facts such as credits or graduation requirements.
        scope = answer_scope(clean_question, evidence) if classify_query(clean_question) == OFFICIAL_FACT else None
        if scope:
            evidence, notice, options = scope
            response = self.generator.answer(clean_question, evidence, recent_history)
            if response.grounded:
                response.scope_notice = notice
                response.scope_options = options or None
                return response
            clarification = clarification_for_versions(clean_question, retrieved_evidence)
            if clarification:
                message, clarification_options = clarification
                return ChatResponse(
                    answer=message,
                    sources=[],
                    grounded=False,
                    needs_clarification=True,
                    clarification_options=clarification_options,
                )
            return response
        return self.generator.answer(clean_question, evidence, recent_history)
