import pytest

from app.config import Settings
from app.models import RetrievedEvidence
from app.retrieval import (
    CampusRetriever,
    deduplicate_evidence,
    distance_to_relevance,
    expand_query,
    keyword_score,
    normalize_search_text,
)


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


def test_search_text_normalizes_traditional_chinese():
    assert normalize_search_text("電子遊戲設計與開發") == "电子游戏设计与开发"


def test_query_expansion_adds_chinese_english_course_aliases():
    game_variants = expand_query("有没有游戏制作课程")
    bio_variants = expand_query("bio相关课程")

    assert "video games design and development" in game_variants
    assert "电子游戏设计与开发" in game_variants
    assert "life sciences" in bio_variants
    assert "生命科学" in bio_variants


def test_cosine_distance_is_converted_to_bounded_relevance():
    assert distance_to_relevance(0.18, "cosine") == pytest.approx(0.82)
    assert distance_to_relevance(1.5, "cosine") == 0.0
    assert distance_to_relevance(-1, "cosine") == 1.0


class FakeDocument:
    def __init__(self, text: str, page: int):
        self.page_content = text
        self.metadata = {"document": "培养方案.pdf", "page": page}


class FakeVectorstore:
    def similarity_search_with_score(self, query: str, k: int):
        assert query == "Visual Analytics 的课程代码是什么？"
        assert k == 4
        return [
            (FakeDocument("Visual Analytics 的课程代码是 DDA3003。", 3), 0.12),
            (FakeDocument("不相关内容", 7), 0.9),
        ]

    def get(self, include):
        return {"documents": [], "metadatas": []}


def test_retriever_uses_raw_cosine_distance_before_thresholding():
    settings = Settings(top_k=2, fetch_k=4, similarity_threshold=0.45, hybrid_search=False)
    results = CampusRetriever(FakeVectorstore(), settings).retrieve("Visual Analytics 的课程代码是什么？")
    assert len(results) == 1
    assert results[0].page == 3
    assert results[0].score == 0.88
