from app.models import RetrievedEvidence
from app.retrieval import deduplicate_evidence, keyword_score


def evidence(text: str, score: float, page: int = 1) -> RetrievedEvidence:
    return RetrievedEvidence(text=text, document="学生手册.pdf", page=page, score=score)


def test_deduplicate_evidence_keeps_highest_score():
    results = deduplicate_evidence(
        [
            evidence("补办 学生证", 0.6),
            evidence("补办   学生证", 0.9),
            evidence("图书馆开放时间", 0.7, page=2),
        ],
        top_k=3,
    )
    assert len(results) == 2
    assert results[0].score == 0.9


def test_keyword_score_prefers_matching_chinese_text():
    assert keyword_score("学生证如何补办", "学生证丢失后可以申请补办") > keyword_score(
        "学生证如何补办", "图书馆开放时间"
    )
