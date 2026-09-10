"""Shared API and internal data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True)
class TextChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedEvidence:
    text: str
    document: str
    page: int | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatHistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=10)


class SourceResponse(BaseModel):
    document: str
    page: int | None = Field(default=None, ge=1)
    excerpt: str
    score: float = Field(ge=0, le=1)
    source_type: str = "pdf"
    url: str | None = None
    section: str | None = None
    crawled_at: str | None = None
    department_zh: str | None = None
    department_en: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    grounded: bool
    needs_clarification: bool | None = None
    clarification_options: list[str] | None = None
    scope_notice: str | None = None
    scope_options: list[str] | None = None
