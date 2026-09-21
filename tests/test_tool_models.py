import pytest
from pydantic import ValidationError

from app.tool_models import (
    CompareSchemeVersionsArgs,
    GetCourseArgs,
    SearchKnowledgeArgs,
    ToolCall,
    ToolEvidence,
    ToolResult,
)


def test_course_code_is_normalized():
    assert GetCourseArgs(course_code=" csc1001 ").course_code == "CSC1001"


def test_invalid_course_code_is_rejected():
    with pytest.raises(ValidationError):
        GetCourseArgs(course_code="not-a-course")


def test_scheme_years_are_sorted_and_unique():
    args = CompareSchemeVersionsArgs(programme="金融学", admission_years=[2024, 2023])
    assert args.admission_years == [2023, 2024]

    with pytest.raises(ValidationError):
        CompareSchemeVersionsArgs(programme="金融学", admission_years=[2023, 2023])


def test_tool_evidence_and_result_have_safe_defaults():
    evidence = ToolEvidence(text="说明", document="学生手册.pdf", page=1, score=0.9)
    result = ToolResult(tool_name="search_knowledge", success=True, evidence=[evidence])
    assert result.data is None
    assert result.error_code is None
    assert result.evidence[0].source_type == "pdf"


def test_tool_call_keeps_name_and_arguments_typed():
    call = ToolCall(name="search_knowledge", arguments={"query": "课程要求"})
    assert call.name == "search_knowledge"
    assert call.arguments["query"] == "课程要求"


def test_search_arguments_limit_result_count():
    with pytest.raises(ValidationError):
        SearchKnowledgeArgs(query="课程", max_results=21)
