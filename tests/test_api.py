from fastapi.testclient import TestClient

from app.api import create_app
from app.models import ChatResponse, SourceResponse


class FakeService:
    def ask(self, question: str, history) -> ChatResponse:
        assert question == "学生证怎么补办？"
        assert history == []
        return ChatResponse(
            answer="请向学生事务中心申请。【学生手册.pdf，第 12 页】",
            sources=[SourceResponse(document="学生手册.pdf", page=12, excerpt="学生证补办流程", score=0.82)],
            grounded=True,
        )


def test_chat_api_response_structure():
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/chat", json={"question": "学生证怎么补办？"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "请向学生事务中心申请。【学生手册.pdf，第 12 页】",
        "sources": [
            {
                "document": "学生手册.pdf",
                "page": 12,
                "excerpt": "学生证补办流程",
                "score": 0.82,
                "source_type": "pdf",
            }
        ],
        "grounded": True,
    }


def test_chat_api_validates_empty_question():
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_api_passes_recent_history_to_service():
    class HistoryService:
        def ask(self, question, history):
            assert question == "还有哪些？"
            assert [(item.role, item.content) for item in history] == [
                ("user", "金融学有哪些必修课？"),
                ("assistant", "目前能确认 FIN2020。"),
            ]
            return ChatResponse(answer="我再帮你看看。", sources=[], grounded=False)

    payload = {
        "question": "还有哪些？",
        "history": [
            {"role": "user", "content": "金融学有哪些必修课？"},
            {"role": "assistant", "content": "目前能确认 FIN2020。"},
        ],
    }
    with TestClient(create_app(service=HistoryService())) as client:
        response = client.post("/chat", json=payload)
    assert response.status_code == 200


def test_chat_api_returns_clarification_fields_when_needed():
    class ClarificationService:
        def ask(self, question, history):
            return ChatResponse(
                answer="我找到了两个版本，你想看哪一版？",
                sources=[],
                grounded=False,
                needs_clarification=True,
                clarification_options=["适用于2023至24年度入学学生", "适用于2024至25年度入学学生"],
            )

    with TestClient(create_app(service=ClarificationService())) as client:
        response = client.post("/chat", json={"question": "金融学需要多少学分？"})

    assert response.status_code == 200
    assert response.json()["needs_clarification"] is True
    assert len(response.json()["clarification_options"]) == 2
