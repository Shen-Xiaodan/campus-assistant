"""Allowlisted tool registration and argument validation."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import ValidationError

from app.tool_models import ToolDefinition, ToolResult
from app.tools import AgentTool


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool[Any]] = ()):
        registered = list(tools)
        names = [tool.name for tool in registered]
        if len(names) != len(set(names)):
            raise ValueError("工具名称不能重复")
        self._tools = {tool.name: tool for tool in registered}

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error_code="unknown_tool",
                error_message="工具未注册",
            )
        try:
            parsed = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error_code="invalid_arguments",
                error_message="工具参数无效",
                data={"fields": exc.errors(include_url=False)},
            )
        try:
            return tool.run(parsed)
        except Exception:
            return ToolResult(
                tool_name=name,
                success=False,
                error_code="execution_failed",
                error_message="工具执行失败",
            )
