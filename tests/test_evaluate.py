from app.evaluate import evaluate


def test_offline_evaluation_metrics():
    dataset = [
        {
            "question": "学生证补办",
            "relevant_document": "手册.pdf",
            "relevant_pages": [2],
            "should_refuse": False,
        },
        {
            "question": "火锅推荐",
            "relevant_document": None,
            "relevant_pages": [],
            "should_refuse": True,
        },
    ]
    corpus = [{"document": "手册.pdf", "page": 2, "text": "学生证补办流程"}]
    metrics = evaluate(dataset, corpus, threshold=0.35)
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["citation_accuracy"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
