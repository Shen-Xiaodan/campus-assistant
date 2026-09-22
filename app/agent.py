"""Bounded, allowlisted tool-calling loop.

The controller is intentionally independent of a specific model provider. A
planner decides what to do, while the registry validates and executes tools.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import TypeAdapter

from app.models import ChatHistoryMessage
from app.tool_models import (
    AgentDecision,
    AgentRunResult,
    AgentStep,
    FinalAnswerDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from app.tool_registry import ToolRegistry

MAX_TOOL_STEPS = 3
_DECISION_ADAPTER = TypeAdapter(AgentDecision)
logger = logging.getLogger(__name__)


class AgentPlanner(Protocol):
    def decide(
        self,
        question: str,
        history: list[ChatHistoryMessage],
        tools: list[ToolDefinition],
        steps: list[AgentStep],
    ) -> AgentDecision:
        ...


def tool_call_fingerprint(tool_call: ToolCall) -> str:
    arguments = json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)
    return f"{tool_call.name}:{arguments}"


class AgentController:
    def __init__(
        self,
        planner: AgentPlanner,
        registry: ToolRegistry,
        max_tool_steps: int = MAX_TOOL_STEPS,
    ):
        if not 1 <= max_tool_steps <= MAX_TOOL_STEPS:
            raise ValueError(f"max_tool_steps 必须位于 1 到 {MAX_TOOL_STEPS} 之间")
        self.planner = planner
        self.registry = registry
        self.max_tool_steps = max_tool_steps

    def run(
        self,
        question: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> AgentRunResult:
        recent_history = (history or [])[-6:]
        steps: list[AgentStep] = []
        fingerprints: set[str] = set()

        for step_number in range(1, self.max_tool_steps + 2):
            try:
                decision = _DECISION_ADAPTER.validate_python(self.planner.decide(
                    question=question,
                    history=recent_history,
                    tools=self.registry.definitions(),
                    steps=steps,
                ))
            except Exception as exc:
                logger.warning("agent_planner_failed step=%d error_type=%s", step_number, type(exc).__name__)
                return AgentRunResult(
                    steps=steps,
                    completed=False,
                    stop_reason="planner_error",
                )

            if isinstance(decision, FinalAnswerDecision):
                return AgentRunResult(
                    answer=decision.answer,
                    steps=steps,
                    completed=True,
                    stop_reason="final_answer",
                )

            if step_number > self.max_tool_steps:
                break

            tool_call = decision.tool_call
            fingerprint = tool_call_fingerprint(tool_call)
            if fingerprint in fingerprints:
                result = ToolResult(
                    tool_name=tool_call.name,
                    success=False,
                    error_code="duplicate_call",
                    error_message="相同工具和参数已经执行过",
                )
            else:
                fingerprints.add(fingerprint)
                result = self.registry.execute(tool_call.name, tool_call.arguments)

            steps.append(
                AgentStep(
                    step_number=step_number,
                    tool_call=tool_call,
                    result=result,
                )
            )
            logger.info(
                "agent_tool_completed step=%d tool=%s success=%s evidence=%d",
                step_number,
                tool_call.name,
                result.success,
                len(result.evidence),
            )

        return AgentRunResult(
            steps=steps,
            completed=False,
            stop_reason="max_steps",
        )
