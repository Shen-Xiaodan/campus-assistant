from app.config import Settings
from app.generation import REFUSAL_ANSWER, AnswerGenerator, citation_label
from app.models import RetrievedEvidence


class FakeModel:
    def invoke(self, prompt: str) -> str:
        assert "学生手册.pdf" in prompt
        return "请到学生事务中心提交申请。【学生手册.pdf，第 12 页】"


def test_citation_format():
    assert citation_label("学生手册.pdf", 12) == "【学生手册.pdf，第 12 页】"


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
