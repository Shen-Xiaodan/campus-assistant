"""Shared API and internal data models."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator


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


class ModelConnection(BaseModel):
    provider: str = Field(default="openai-compatible", pattern="^(siliconflow|openai|openai-compatible)$")
    api_key: SecretStr = Field(min_length=1, max_length=1000)
    base_url: str = Field(min_length=1, max_length=500)
    model_id: str = Field(min_length=1, max_length=200)

    @field_validator("base_url")
    @classmethod
    def validate_public_https_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("模型 API 地址必须是有效的 HTTPS 地址")
        hostname = parsed.hostname.lower()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("模型 API 地址不能指向本机或私有网络")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError("模型 API 地址不能指向本机或私有网络")
        return normalized

    @field_validator("model_id")
    @classmethod
    def strip_model_id(cls, value: str) -> str:
        return value.strip()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=10)
    model: ModelConnection


class ModelCheckRequest(BaseModel):
    model: ModelConnection


class ModelCheckResponse(BaseModel):
    connected: bool
    message: str


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
    answer_mode: str | None = None
    disclaimer: str | None = None
    needs_clarification: bool | None = None
    clarification_options: list[str] | None = None
    scope_notice: str | None = None
    scope_options: list[str] | None = None
