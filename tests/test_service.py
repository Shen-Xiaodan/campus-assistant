from app.models import ChatHistoryMessage, ChatResponse, RetrievedEvidence
from app.service import QAService, clarification_for_versions


class RecordingRetriever:
    def __init__(self):
        self.query = ""

    def retrieve(self, query):
        self.query = query
        return []


class RecordingGenerator:
    def __init__(self):
        self.history = []

    def answer(self, question, evidence, history):
        self.history = history
        return ChatResponse(answer="暂未找到。", sources=[], grounded=False)


def test_service_uses_recent_context_for_follow_up_retrieval_and_generation():
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    service = QAService(retriever, generator)
    history = [
        ChatHistoryMessage(role="user", content="金融学有哪些必修课？"),
        ChatHistoryMessage(role="assistant", content="目前能确认 FIN2020。"),
    ]

    service.ask("还有呢？", history)

    assert "金融学有哪些必修课？" in retriever.query
    assert retriever.query.endswith("还有呢？")
    assert generator.history == history


def test_clarification_is_requested_for_competing_programme_versions():
    evidence = [
        RetrievedEvidence("2023版要求", "金融学_适用于2023至24年度入学学生.pdf", 1, 0.9),
        RetrievedEvidence("2024版要求", "金融学_适用于2024至25年度入学学生.pdf", 1, 0.88),
    ]
    message, options = clarification_for_versions("金融学需要多少学分？", evidence) or ("", [])

    assert "多个版本" in message
    assert options == ["适用于2023至24年度入学学生", "适用于2024至25年度入学学生"]


def test_clarification_is_skipped_when_year_is_already_present():
    evidence = [
        RetrievedEvidence("2023版要求", "金融学_适用于2023至24年度入学学生.pdf", 1, 0.9),
        RetrievedEvidence("2024版要求", "金融学_适用于2024至25年度入学学生.pdf", 1, 0.88),
    ]
    assert clarification_for_versions("2023至24年度金融学需要多少学分？", evidence) is None


def test_one_document_covering_multiple_years_does_not_trigger_clarification():
    evidence = [
        RetrievedEvidence(
            "共同要求",
            "金融工程_适用于2021至22、2022至23、2023至24及2024至25年度入学学生.pdf",
            1,
            0.9,
        )
    ]
    assert clarification_for_versions("金融工程需要多少学分？", evidence) is None


def test_service_returns_clarification_without_calling_generator():
    class VersionRetriever:
        def retrieve(self, query):
            return [
                RetrievedEvidence("旧版", "金融学_适用于2023至24年度入学学生.pdf", 1, 0.9),
                RetrievedEvidence("新版", "金融学_适用于2024至25年度入学学生.pdf", 1, 0.8),
            ]

    class GeneratorMustNotRun:
        def answer(self, *args):
            raise AssertionError("ambiguous evidence should be clarified before generation")

    result = QAService(VersionRetriever(), GeneratorMustNotRun()).ask("金融学需要多少学分？")
    assert result.needs_clarification is True
    assert len(result.clarification_options) == 2
    assert result.sources == []
