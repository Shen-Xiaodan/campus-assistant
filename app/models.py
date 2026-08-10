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
    page: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    document: str
    page: int = Field(ge=1)
    excerpt: str
    score: float = Field(ge=0, le=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    grounded: bool
