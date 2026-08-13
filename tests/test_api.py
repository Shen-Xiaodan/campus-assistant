from fastapi.testclient import TestClient

from app.api import create_app
from app.models import ChatResponse, SourceResponse

MODEL = {
    "provider": "openai-compatible",
    "api_key": "test-key",
    "base_url": "https://api.example.com/v1",
    "model_id": "test-model",
}


class FakeService:
    def ask(self, question: str, history, connection) -> ChatResponse:
        assert question == "学生证怎么补办？"
        assert history == []
        assert connection.api_key.get_secret_value() == "test-key"
        return ChatResponse(
            answer="请向学生事务中心申请。【学生手册.pdf，第 12 页】",
            sources=[SourceResponse(document="学生手册.pdf", page=12, excerpt="学生证补办流程", score=0.82)],
            grounded=True,
        )


def test_chat_api_response_structure():
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/chat", json={"question": "学生证怎么补办？", "model": MODEL})
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
        response = client.post("/chat", json={"question": "", "model": MODEL})
    assert response.status_code == 422


def test_chat_api_passes_recent_history_to_service():
    class HistoryService:
        def ask(self, question, history, connection):
            assert question == "还有哪些？"
            assert [(item.role, item.content) for item in history] == [
                ("user", "金融学有哪些必修课？"),
                ("assistant", "目前能确认 FIN2020。"),
            ]
            return ChatResponse(answer="我再帮你看看。", sources=[], grounded=False)

    payload = {
        "question": "还有哪些？",
        "model": MODEL,
        "history": [
            {"role": "user", "content": "金融学有哪些必修课？"},
            {"role": "assistant", "content": "目前能确认 FIN2020。"},
        ],
    }
    with TestClient(create_app(service=HistoryService())) as client:
        response = client.post("/chat", json=payload)
    assert response.status_code == 200


def test_chat_api_requires_user_model_configuration():
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/chat", json={"question": "学生证怎么补办？"})
    assert response.status_code == 422


def test_model_check_uses_request_credentials(monkeypatch):
    captured = {}

    def fake_check(connection):
        captured["key"] = connection.api_key.get_secret_value()
        captured["model"] = connection.model_id

    monkeypatch.setattr("app.api.check_model_connection", fake_check)
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/model/check", json={"model": MODEL})

    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert captured == {"key": "test-key", "model": "test-model"}


def test_model_configuration_rejects_private_network_url():
    private_model = {**MODEL, "base_url": "https://127.0.0.1/v1"}
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/model/check", json={"model": private_model})
    assert response.status_code == 422
