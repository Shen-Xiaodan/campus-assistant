"""Protocols and shared helpers for read-only agent tools.

Concrete tool implementations will be added in the next phase. Keeping the
protocol here lets the registry and tests stay independent of any LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from app.models import RetrievedEvidence
from app.tool_models import ToolDefinition, ToolEvidence, ToolResult

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class AgentTool(Protocol[ArgsT]):
    """A typed, explicitly registered tool callable."""

    name: str
    description: str
    args_model: type[ArgsT]

    def definition(self) -> ToolDefinition: ...

    def run(self, arguments: ArgsT) -> ToolResult: ...


class BaseTool(Generic[ArgsT], ABC):
    """Convenience base class for concrete tools."""

    name: str
    description: str
    args_model: type[ArgsT]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )

    @abstractmethod
    def run(self, arguments: ArgsT) -> ToolResult:
        raise NotImplementedError


def evidence_to_tool_evidence(evidence: RetrievedEvidence) -> ToolEvidence:
    metadata = dict(evidence.metadata)
    return ToolEvidence(
        text=evidence.text,
        document=evidence.document,
        page=evidence.page,
        score=evidence.score,
        source_type=str(metadata.get("source_type", "pdf")),
        url=metadata.get("source_url"),
        section=metadata.get("section"),
        metadata={
            key: value
            for key, value in metadata.items()
            if key in {"department", "crawled_at"}
        },
    )
