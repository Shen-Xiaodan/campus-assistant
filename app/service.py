"""Application service composing retrieval and generation."""

from __future__ import annotations

from typing import Protocol

from app.generation import AnswerGenerator
from app.models import ChatResponse, RetrievedEvidence


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...


class QAService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str) -> ChatResponse:
        clean_question = question.strip()
        evidence = self.retriever.retrieve(clean_question)
        return self.generator.answer(clean_question, evidence)
