from app.models import ChatHistoryMessage, ChatResponse
from app.service import QAService


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
