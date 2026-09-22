from fastapi.testclient import TestClient

from app.agent_response import answer_from_run
from app.agent_service import AgentQAService, should_use_agent
from app.api import create_app
from app.config import Settings
from app.generation import AnswerGenerator
from app.models import ChatResponse, RetrievedEvidence
from app.tool_models import AgentRunResult, AgentStep, ToolCall, ToolEvidence, ToolResult


class FakeFallback:
    def __init__(self):
        self.calls = 0

    def ask(self, question, history):
        self.calls += 1
        return ChatResponse(answer="原有回答", sources=[], grounded=False)


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def run(self, question, history):
        self.calls += 1
        return self.result


class FakeGenerator:
    def answer(self, question, evidence, history):
        assert evidence[0].document == "培养方案.pdf"
        return ChatResponse(answer="有变化。【培养方案.pdf，第 2 页】", sources=[], grounded=True)


def successful_run():
    return AgentRunResult(
        completed=True,
        stop_reason="final_answer",
        answer="未经校验的模型总结",
        steps=[AgentStep(
            step_number=1,
            tool_call=ToolCall(name="search_knowledge", arguments={"query": "要求"}),
            result=ToolResult(
                tool_name="search_knowledge",
                success=True,
                evidence=[ToolEvidence(text="要求", document="培养方案.pdf", page=2, score=0.8)],
            ),
        )],
    )


def test_complex_question_uses_agent_evidence_to_generate_answer():
    fallback = FakeFallback()
    agent = FakeAgent(successful_run())
    service = AgentQAService(agent, FakeGenerator(), fallback)
    response = service.ask("比较 2023 和 2024 年培养方案", [])
    assert response.grounded is True
    assert response.answer != "未经校验的模型总结"
    assert agent.calls == 1
    assert fallback.calls == 0


def test_simple_question_and_agent_failure_use_existing_service():
    fallback = FakeFallback()
    agent = FakeAgent(AgentRunResult(completed=False, stop_reason="planner_error"))
    service = AgentQAService(agent, FakeGenerator(), fallback)
    assert should_use_agent("CSC1001多少学分？") is False
    assert service.ask("CSC1001多少学分？", []).answer == "原有回答"
    assert agent.calls == 0
    assert service.ask("比较 2023 和 2024 年培养方案", []).answer == "原有回答"
    assert fallback.calls == 2


def test_agent_without_citable_evidence_cannot_answer():
    run = successful_run()
    run.steps[0].result.evidence = []
    assert answer_from_run(run, "比较 2023 和 2024", [], FakeGenerator()) is None


def test_planner_failure_after_successful_tool_uses_existing_evidence():
    run = successful_run()
    run.completed = False
    run.stop_reason = "planner_error"
    response = answer_from_run(run, "比较 2023 和 2024", [], FakeGenerator())
    assert response is not None and response.grounded is True


def test_missing_comparison_version_reports_gap_without_generating():
    run = successful_run()
    run.steps[0].tool_call.name = "compare_scheme_versions"
    run.steps[0].result.data = {
        "versions": [
            {"admission_year": 2023, "evidence_indexes": [0]},
            {"admission_year": 2024, "evidence_indexes": []},
        ]
    }
    class NoGenerator:
        settings = type("Settings", (), {"max_excerpt_chars": 240})()

        def answer(self, question, evidence, history):
            raise AssertionError("incomplete versions should not invoke the model")

    response = answer_from_run(run, "比较 2023 和 2024", [], NoGenerator())
    assert response is not None
    assert "未找到 2024" in response.answer
    assert response.grounded is False
    assert len(response.sources) == 1


def test_chat_api_uses_built_agent_for_comparison(monkeypatch, tmp_path):
    (tmp_path / "index_manifest.json").write_text('{"documents":{"sample":{}}}', encoding="utf-8")

    class Retriever:
        def __init__(self, vectorstore, settings):
            pass

        def retrieve(self, query):
            return [RetrievedEvidence("2024版要求", "金融学_适用于2024至25年度入学学生.pdf", 2, 0.9)]

        def find_course(self, course_code):
            return []

    class Model:
        def __init__(self):
            self.decisions = 0

        def invoke(self, prompt):
            if "剩余工具次数" in prompt:
                self.decisions += 1
                if self.decisions == 1:
                    return (
                        '{"action":"call_tool","tool_call":'
                        '{"name":"search_knowledge","arguments":{"query":"金融学培养方案"}}}'
                    )
                return '{"action":"final","answer":"已取得资料"}'
            return "2024版有相关要求。【金融学_适用于2024至25年度入学学生.pdf，第 2 页】"

    model = Model()
    monkeypatch.setattr("app.api.open_vectorstore", lambda settings: object())
    monkeypatch.setattr("app.api.CampusRetriever", Retriever)
    monkeypatch.setattr(AnswerGenerator, "_get_model", lambda self: model)
    with TestClient(create_app(settings=Settings(data_dir=tmp_path, index_dir=tmp_path))) as client:
        response = client.post("/chat", json={"question": "比较 2023 和 2024 年金融学培养方案"})
    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["sources"][0]["page"] == 2
    assert model.decisions == 2
