"""Route suitable questions through the bounded agent with RAG fallback."""

from __future__ import annotations

import logging
import re

from app.agent import AgentController
from app.agent_response import answer_from_run
from app.generation import AnswerGenerator
from app.models import ChatHistoryMessage, ChatResponse
from app.service import QAService

logger = logging.getLogger(__name__)
_COMPARISON = re.compile(r"比较|对比|区别|变化|差异|compare|difference", re.I)
_YEAR = re.compile(r"20\d{2}")


def should_use_agent(question: str) -> bool:
    """Start with version comparisons that the available read-only tools support."""
    return bool(_COMPARISON.search(question) and len(set(_YEAR.findall(question))) >= 2)


class AgentQAService:
    def __init__(self, agent: AgentController, generator: AnswerGenerator, fallback: QAService):
        self.agent = agent
        self.generator = generator
        self.fallback = fallback

    def ask(self, question: str, history: list[ChatHistoryMessage] | None = None) -> ChatResponse:
        if not should_use_agent(question):
            return self.fallback.ask(question, history)
        try:
            run = self.agent.run(question, history)
            response = answer_from_run(run, question, history, self.generator)
            if response is not None:
                logger.info("agent_completed tools=%s", [step.tool_call.name for step in run.steps])
                return response
            logger.info("agent_fallback reason=%s", run.stop_reason)
        except Exception:
            logger.exception("agent_failed")
        return self.fallback.ask(question, history)
