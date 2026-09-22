from pydantic import BaseModel

from app.agent import AgentController
from app.models import ChatHistoryMessage
from app.tool_models import FinalAnswerDecision, ToolCall, ToolDefinition, ToolResult
from app.tool_registry import ToolRegistry


class EchoArgs(BaseModel):
    value: str


class EchoTool:
    name = "echo"
    description = "测试工具"
    args_model = EchoArgs

    def definition(self):
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )

    def run(self, arguments):
        return ToolResult(tool_name=self.name, success=True, data={"value": arguments.value})


class SequencePlanner:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.seen_steps = []

    def decide(self, question, history, tools, steps):
        self.seen_steps.append((question, history, tools, list(steps)))
        return next(self.decisions)


def controller(planner):
    return AgentController(planner, ToolRegistry([EchoTool()]))


def call(value="ok"):
    from app.tool_models import ToolCallDecision

    return ToolCallDecision(action="call_tool", tool_call=ToolCall(name="echo", arguments={"value": value}))


def test_agent_finishes_after_tool_call_and_final_decision():
    planner = SequencePlanner([call(), FinalAnswerDecision(action="final", answer="完成")])
    result = controller(planner).run(
        "测试",
        [ChatHistoryMessage(role="user", content="上文")],
    )
    assert result.completed is True
    assert result.answer == "完成"
    assert result.stop_reason == "final_answer"
    assert len(result.steps) == 1
    assert len(planner.seen_steps[1][3]) == 1


def test_agent_stops_after_three_tool_calls():
    planner = SequencePlanner([call("1"), call("2"), call("3"), FinalAnswerDecision(action="final", answer="不应调用")])
    result = controller(planner).run("测试")
    assert result.completed is True
    assert result.stop_reason == "final_answer"
    assert len(result.steps) == 3
    assert len(planner.seen_steps) == 4


def test_agent_refuses_fourth_tool_call():
    planner = SequencePlanner([call("1"), call("2"), call("3"), call("4")])
    result = controller(planner).run("测试")
    assert result.completed is False
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3


def test_agent_rejects_duplicate_tool_calls_without_executing_again():
    planner = SequencePlanner([call(), call(), FinalAnswerDecision(action="final", answer="完成")])
    result = controller(planner).run("测试")
    assert result.steps[0].result.success is True
    assert result.steps[1].result.error_code == "duplicate_call"


def test_agent_preserves_registry_errors_for_unknown_tool():
    from app.tool_models import ToolCallDecision

    planner = SequencePlanner([
        ToolCallDecision(action="call_tool", tool_call=ToolCall(name="shell", arguments={})),
        FinalAnswerDecision(action="final", answer="无法执行"),
    ])
    result = controller(planner).run("测试")
    assert result.steps[0].result.error_code == "unknown_tool"


def test_agent_converts_planner_error_to_safe_result():
    class BrokenPlanner:
        def decide(self, **kwargs):
            raise RuntimeError("内部模型响应")

    result = controller(BrokenPlanner()).run("测试")
    assert result.completed is False
    assert result.stop_reason == "planner_error"
    assert result.steps == []


def test_agent_limits_history_to_six_messages():
    planner = SequencePlanner([FinalAnswerDecision(action="final", answer="完成")])
    history = [ChatHistoryMessage(role="user", content=str(index)) for index in range(8)]
    controller(planner).run("测试", history)
    assert [item.content for item in planner.seen_steps[0][1]] == ["2", "3", "4", "5", "6", "7"]
