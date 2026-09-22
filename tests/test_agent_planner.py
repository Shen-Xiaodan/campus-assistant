import pytest

from app.agent_planner import AgentPlannerError, LLMToolPlanner, parse_agent_decision
from app.tool_models import AgentStep, ToolCall, ToolDefinition, ToolEvidence, ToolResult


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return self.response


def test_planner_reads_tool_decision_and_definitions():
    model = FakeModel('{"action":"call_tool","tool_call":{"name":"search_knowledge","arguments":{"query":"学分"}}}')
    planner = LLMToolPlanner(model)
    decision = planner.decide("学分？", [], [ToolDefinition(name="search_knowledge", description="检索")], [])
    assert decision.tool_call.arguments == {"query": "学分"}
    assert "search_knowledge" in model.prompt


def test_planner_parses_fenced_final_and_rejects_invalid_output():
    assert parse_agent_decision('```json\n{"action":"final","answer":"完成"}\n```').answer == "完成"
    assert parse_agent_decision('<think>分析</think>\n{"action":"final","answer":"完成"}').answer == "完成"
    for output in ('{"action":"unknown"}', 'before {"action":"final","answer":"完成"}', "not json"):
        with pytest.raises(AgentPlannerError):
            parse_agent_decision(output)


def test_planner_finishes_after_successful_version_comparison_without_model_call():
    model = FakeModel("this response should never be used")
    step = AgentStep(
        step_number=1,
        tool_call=ToolCall(name="compare_scheme_versions", arguments={}),
        result=ToolResult(
            tool_name="compare_scheme_versions",
            success=True,
            evidence=[ToolEvidence(text="要求", document="培养方案.pdf", page=1, score=0.9)],
        ),
    )
    decision = LLMToolPlanner(model).decide("比较", [], [], [step])
    assert decision.action == "final"
    assert model.prompt == ""


def test_planner_retries_one_invalid_model_decision():
    class RepairModel:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "invalid"
            return '{"action":"final","answer":"完成"}'

    model = RepairModel()
    decision = LLMToolPlanner(model).decide("测试", [], [], [])
    assert decision.action == "final"
    assert model.calls == 2
