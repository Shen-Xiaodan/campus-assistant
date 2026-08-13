import pytest

from app.config import Settings
from app.models import RetrievedEvidence
from app.retrieval import (
    CampusRetriever,
    deduplicate_evidence,
    distance_to_relevance,
    diversify_evidence,
    expand_query,
    is_aggregate_query,
    keyword_score,
    matching_study_scheme_documents,
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


def test_query_expansion_supports_common_bilingual_campus_terms():
    english_variants = expand_query("Where can I replace my student card?")
    chinese_variants = expand_query("奖学金申请条件是什么？")

    assert "学生证" in english_variants
    assert "校园卡" in english_variants
    assert "scholarship" in chinese_variants
    assert "financial aid" in chinese_variants


def test_aggregate_query_detection_is_high_precision():
    assert is_aggregate_query("港中深有哪些专业？")
    assert is_aggregate_query("请列出学校全部课程")
    assert is_aggregate_query("List all available programmes")
    assert not is_aggregate_query("申请奖学金需要满足哪些条件？")
    assert not is_aggregate_query("Visual Analytics 的课程代码是什么？")


def test_explicit_programme_query_matches_only_its_study_scheme():
    metadatas = [
        {"document": "金融学_适用于2023至24年度入学学生.pdf"},
        {"document": "金融工程_适用于2023至24年度入学学生.pdf"},
        {"document": "数据科学与大数据技术_适用于2023至24年度入学学生.pdf"},
    ]
    assert matching_study_scheme_documents("金融学的必修课", metadatas) == {
        "金融学_适用于2023至24年度入学学生.pdf"
    }
    assert matching_study_scheme_documents("金融工程有哪些必修课", metadatas) == {
        "金融工程_适用于2023至24年度入学学生.pdf"
    }
    assert matching_study_scheme_documents("港中深有哪些专业", metadatas) == set()


def test_diversify_evidence_prefers_document_coverage_before_second_chunks():
    results = diversify_evidence(
        [
            RetrievedEvidence("甲-1", "甲.pdf", 1, 0.99),
            RetrievedEvidence("甲-2", "甲.pdf", 2, 0.98),
            RetrievedEvidence("乙-1", "乙.pdf", 1, 0.80),
            RetrievedEvidence("丙-1", "丙.pdf", 1, 0.70),
        ],
        top_k=3,
        max_chunks_per_document=2,
    )
    assert [item.document for item in results] == ["甲.pdf", "乙.pdf", "丙.pdf"]


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


class AggregateVectorstore:
    def similarity_search_with_score(self, query: str, k: int):
        assert query == "港中深有哪些专业？"
        assert k == 8
        return [
            (FakeDocument("专业甲的培养方案", 1), 0.05),
            (FakeDocument("专业甲的课程设置", 2), 0.06),
            (
                type(
                    "Document",
                    (),
                    {"page_content": "专业乙的培养方案", "metadata": {"document": "专业乙.pdf", "page": 1}},
                )(),
                0.20,
            ),
            (
                type(
                    "Document",
                    (),
                    {"page_content": "专业丙的培养方案", "metadata": {"document": "专业丙.pdf", "page": 1}},
                )(),
                0.25,
            ),
        ]

    def get(self, include):
        return {"documents": [], "metadatas": []}


def test_aggregate_query_uses_wider_fetch_and_document_diversity():
    settings = Settings(
        top_k=2,
        fetch_k=4,
        aggregate_top_k=3,
        aggregate_fetch_k=8,
        aggregate_max_chunks_per_document=1,
        similarity_threshold=0.4,
        hybrid_search=False,
    )
    results = CampusRetriever(AggregateVectorstore(), settings).retrieve("港中深有哪些专业？")
    assert [item.document for item in results] == ["培养方案.pdf", "专业乙.pdf", "专业丙.pdf"]


class ProgrammeVectorstore:
    def similarity_search_with_score(self, query: str, k: int):
        return [
            (
                type(
                    "Document",
                    (),
                    {
                        "page_content": "金融学必修科目 FIN2020",
                        "metadata": {"document": "金融学_适用于2023至24年度入学学生.pdf", "page": 2},
                    },
                )(),
                0.1,
            ),
            (
                type(
                    "Document",
                    (),
                    {
                        "page_content": "数据科学必修科目 DDA4002",
                        "metadata": {
                            "document": "数据科学与大数据技术_适用于2023至24年度入学学生.pdf",
                            "page": 1,
                        },
                    },
                )(),
                0.11,
            ),
        ]

    def get(self, include):
        return {
            "documents": ["金融学必修科目 FIN2020", "数据科学必修科目 DDA4002"],
            "metadatas": [
                {"document": "金融学_适用于2023至24年度入学学生.pdf", "page": 2},
                {"document": "数据科学与大数据技术_适用于2023至24年度入学学生.pdf", "page": 1},
            ],
        }


def test_explicit_programme_query_filters_other_programme_documents():
    settings = Settings(top_k=4, fetch_k=4, similarity_threshold=0.4, hybrid_search=False)
    results = CampusRetriever(ProgrammeVectorstore(), settings).retrieve("金融学的必修课")
    assert len(results) == 1
    assert results[0].document.startswith("金融学_")
