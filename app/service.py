"""Application service composing retrieval and generation."""

from __future__ import annotations

from typing import Protocol

from app.generation import AnswerGenerator
from app.models import ChatHistoryMessage, ChatResponse, RetrievedEvidence


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...


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
        return self.generator.answer(clean_question, evidence, recent_history)
