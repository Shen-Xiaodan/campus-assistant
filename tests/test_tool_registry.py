from pydantic import BaseModel

from app.tool_models import ToolDefinition, ToolResult
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


def test_registry_validates_and_executes_allowlisted_tools():
    registry = ToolRegistry([EchoTool()])
    result = registry.execute("echo", {"value": "ok"})
    assert result.success is True
    assert result.data == {"value": "ok"}
    assert registry.definitions()[0].name == "echo"


def test_registry_rejects_unknown_tools_and_invalid_arguments():
    registry = ToolRegistry([EchoTool()])
    unknown = registry.execute("shell", {})
    invalid = registry.execute("echo", {"value": 1})
    assert unknown.error_code == "unknown_tool"
    assert invalid.error_code == "invalid_arguments"


def test_registry_rejects_duplicate_tool_names():
    try:
        ToolRegistry([EchoTool(), EchoTool()])
    except ValueError as exc:
        assert "不能重复" in str(exc)
    else:
        raise AssertionError("duplicate tool names should fail")
