"""Protocols and shared helpers for read-only agent tools.

Concrete tool implementations will be added in the next phase. Keeping the
protocol here lets the registry and tests stay independent of any LLM.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.tool_models import ToolDefinition, ToolResult

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class AgentTool(Protocol[ArgsT]):
    """A typed, explicitly registered tool callable."""

    name: str
    description: str
    args_model: type[ArgsT]

    def definition(self) -> ToolDefinition: ...

    def run(self, arguments: ArgsT) -> ToolResult: ...
