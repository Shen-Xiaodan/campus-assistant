from app.config import Settings
from app.generation import REFUSAL_ANSWER, AnswerGenerator, OpenAICompatibleChatModel, citation_label
from app.models import RetrievedEvidence


class FakeModel:
    def invoke(self, prompt: str) -> str:
        assert "学生手册.pdf" in prompt
        return "请到学生事务中心提交申请。【学生手册.pdf，第 12 页】"


def test_citation_format():
    assert citation_label("学生手册.pdf", 12) == "【学生手册.pdf，第 12 页】"
    assert citation_label("图书馆官网", None, "开放时间") == "【图书馆官网，开放时间】"


def test_low_relevance_empty_evidence_refuses_without_model_call():
    result = AnswerGenerator(Settings(), model=FakeModel()).answer("未知问题", [])
    assert result.answer == REFUSAL_ANSWER
    assert result.grounded is False
    assert result.sources == []


def test_grounded_answer_returns_bounded_sources():
    item = RetrievedEvidence("学生证丢失后到学生事务中心提交申请。" * 20, "学生手册.pdf", 12, 0.82)
    result = AnswerGenerator(Settings(max_excerpt_chars=30), model=FakeModel()).answer("怎么补办学生证？", [item])
    assert result.grounded is True
    assert result.sources[0].document == "学生手册.pdf"
    assert result.sources[0].page == 12
    assert len(result.sources[0].excerpt) <= 30


def test_web_evidence_returns_url_and_section():
    class WebModel:
        def invoke(self, prompt: str) -> str:
            assert "【图书馆官网，开放时间】" in prompt
            return "图书馆周一开放。【图书馆官网，开放时间】"

    item = RetrievedEvidence(
        "页面：图书馆官网\n章节：开放时间\n周一开放。",
        "图书馆官网",
        None,
        0.81,
        {
            "source_type": "web",
            "source_url": "https://library.example.edu.cn/hours",
            "section": "开放时间",
            "crawled_at": "2026-08-12T00:00:00+00:00",
        },
    )
    result = AnswerGenerator(Settings(), model=WebModel()).answer("图书馆何时开放？", [item])

    assert result.sources[0].page is None
    assert result.sources[0].source_type == "web"
    assert result.sources[0].url == "https://library.example.edu.cn/hours"
    assert result.sources[0].section == "开放时间"


def test_openai_compatible_provider_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "基于资料的回答"}}]}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("app.generation.requests.post", fake_post)
    model = OpenAICompatibleChatModel("secret-not-printed", "https://api.siliconflow.cn/v1", "demo/model", 30)
    assert model.invoke("校园问题") == "基于资料的回答"
    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["json"]["model"] == "demo/model"
    assert captured["json"]["messages"][0]["content"] == "校园问题"
    assert captured["timeout"] == 30
