from app.knowledge_tools import (
    CompareSchemeVersionsTool,
    GetCourseTool,
    SearchKnowledgeTool,
    build_knowledge_tool_registry,
)
from app.models import RetrievedEvidence
from app.tool_models import CompareSchemeVersionsArgs, GetCourseArgs, SearchKnowledgeArgs


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [
            RetrievedEvidence("要求内容", "金融学_适用于2024至25年度入学学生.pdf", 1, 0.9),
            RetrievedEvidence("官网说明", "教务处官网", None, 0.7, {"source_type": "web", "source_url": "https://example.edu"}),
        ]

    def find_course(self, course_code):
        self.queries.append(course_code)
        return [RetrievedEvidence("CSC1001 课程介绍 3", "课程目录.pdf", 2, 1.0, {"source_type": "course_catalog"})]


def test_search_tool_adds_optional_scope_and_limits_results():
    retriever = FakeRetriever()
    result = SearchKnowledgeTool(retriever).run(
        SearchKnowledgeArgs(query="毕业要求", programme="金融学", admission_year=2024, max_results=1)
    )
    assert "专业：金融学" in retriever.queries[0]
    assert "入学年份：2024" in retriever.queries[0]
    assert result.data["result_count"] == 1


def test_course_tool_normalizes_code_and_preserves_source_type():
    retriever = FakeRetriever()
    result = GetCourseTool(retriever).run(GetCourseArgs(course_code="csc1001"))
    assert retriever.queries == ["CSC1001"]
    assert result.evidence[0].source_type == "course_catalog"


def test_compare_tool_groups_versions_and_filters_other_versions():
    retriever = FakeRetriever()
    result = CompareSchemeVersionsTool(retriever).run(
        CompareSchemeVersionsArgs(programme="金融学", admission_years=[2024, 2025])
    )
    assert len(result.data["versions"]) == 2
    assert "金融学_适用于2024至25年度入学学生.pdf" in result.data["versions"][0]["documents"]


def test_registry_contains_three_knowledge_tools():
    registry = build_knowledge_tool_registry(FakeRetriever())
    assert {item.name for item in registry.definitions()} == {
        "search_knowledge",
        "get_course",
        "compare_scheme_versions",
    }
