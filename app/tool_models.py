"""Typed contracts shared by the read-only tool layer."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator


class ToolEvidence(BaseModel):
    """Evidence returned by a tool and safe to pass to a future agent."""

    text: str
    document: str
    page: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0, le=1)
    source_type: str = "pdf"
    url: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Uniform result envelope for successful and failed tool calls."""

    tool_name: str
    success: bool
    data: dict[str, Any] | list[Any] | None = None
    evidence: list[ToolEvidence] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    @field_validator("error_code", "error_message")
    @classmethod
    def errors_require_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class ToolDefinition(BaseModel):
    """Public description that can later be converted to a model tool schema."""

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Validated tool selection emitted by a future agent planner."""

    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallDecision(BaseModel):
    action: Literal["call_tool"]
    tool_call: ToolCall


class FinalAnswerDecision(BaseModel):
    action: Literal["final"]
    answer: str = Field(min_length=1, max_length=12000)


AgentDecision = Annotated[
    ToolCallDecision | FinalAnswerDecision,
    Field(discriminator="action"),
]


class AgentStep(BaseModel):
    step_number: int = Field(ge=1, le=3)
    tool_call: ToolCall
    result: ToolResult


class AgentRunResult(BaseModel):
    answer: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    completed: bool
    stop_reason: Literal["final_answer", "max_steps", "planner_error", "tool_error"]


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    programme: str | None = Field(default=None, max_length=200)
    admission_year: int | None = Field(default=None, ge=2000, le=2100)
    max_results: int = Field(default=6, ge=1, le=20)


class GetCourseArgs(BaseModel):
    course_code: str = Field(
        min_length=5,
        max_length=10,
        pattern=r"^[A-Za-z]{2,4}\d{4}[A-Za-z]?$",
    )

    @field_validator("course_code", mode="before")
    @classmethod
    def normalize_course_code(cls, value: str) -> str:
        return value.strip().upper()


class CompareSchemeVersionsArgs(BaseModel):
    programme: str = Field(min_length=1, max_length=200)
    admission_years: list[int] = Field(min_length=2, max_length=4)
    query: str | None = Field(default=None, max_length=1000)

    @field_validator("admission_years")
    @classmethod
    def validate_admission_years(cls, value: list[int]) -> list[int]:
        if any(year < 2000 or year > 2100 for year in value):
            raise ValueError("入学年份必须位于 2000 到 2100 之间")
        if len(set(value)) != len(value):
            raise ValueError("入学年份不能重复")
        return sorted(value)


class GetGraduationReportArgs(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=100)
    programme: str = Field(min_length=1, max_length=200)
    admission_year: int = Field(ge=2000, le=2100)
