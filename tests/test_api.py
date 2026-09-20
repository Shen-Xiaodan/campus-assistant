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


def test_chat_api_returns_answer_scope_when_available():
    class ScopedService:
        def ask(self, question, history):
            return ChatResponse(
                answer="需要修满 120 学分。",
                sources=[],
                grounded=True,
                scope_notice="目前按适用于2024至25年度入学学生的金融学培养方案回答。",
                scope_options=["适用于2023至24年度入学学生"],
            )

    with TestClient(create_app(service=ScopedService())) as client:
        response = client.post("/chat", json={"question": "金融学需要多少学分？"})

    assert response.status_code == 200
    assert "2024至25" in response.json()["scope_notice"]
    assert response.json()["scope_options"] == ["适用于2023至24年度入学学生"]


def test_graduation_report_endpoint_returns_json_without_native_table_conversion():
    app = create_app(service=FakeService())
    with TestClient(app) as client:
        app.state.transcripts["test-analysis"] = {
            "programme": "Computer Science and Engineering",
            "admission_year": 2023,
            "courses": [
                {
                    "course_code": "CSC1001",
                    "course_name": "Introduction to Computer Science",
                    "credits": 3,
                    "grade": "A",
                    "term": None,
                    "status": "passed",
                    "source_page": 1,
                    "confidence": 0.9,
                }
            ],
            "total_credits_reported": None,
            "warnings": [],
        }
        response = client.post(
            "/graduation/check",
            json={
                "analysis_id": "test-analysis",
                "programme": "Computer Science and Engineering",
                "admission_year": 2023,
            },
        )

    assert response.status_code == 200
    assert response.json()["programme"] == "计算机科学与技术"
    assert response.json()["detected_programme"] == "Computer Science and Engineering"
    assert response.json()["credits"] == {"earned": 3.0, "required": 70.0, "remaining": 67.0}
    assert response.json()["overall_status"] == "partially_satisfied"
