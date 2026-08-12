from fastapi.testclient import TestClient

from app.api import create_app
from app.models import ChatResponse, SourceResponse


class FakeService:
    def ask(self, question: str) -> ChatResponse:
        assert question == "学生证怎么补办？"
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
