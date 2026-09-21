"""Read-only tools backed by the existing campus retrieval services."""

from __future__ import annotations

import re
from typing import Protocol

from app.models import RetrievedEvidence
from app.retrieval import CampusRetriever
from app.tool_models import (
    CompareSchemeVersionsArgs,
    GetCourseArgs,
    SearchKnowledgeArgs,
    ToolResult,
)
from app.tool_registry import ToolRegistry
from app.tools import BaseTool, evidence_to_tool_evidence

_APPLICABILITY_PATTERN = re.compile(r"^(?P<programme>.+)_适用于(?P<label>.+?)\.pdf$", re.I)


class RetrieverLike(Protocol):
    def retrieve(self, query: str) -> list[RetrievedEvidence]: ...

    def find_course(self, course_code: str) -> list[RetrievedEvidence]: ...


class SearchKnowledgeTool(BaseTool[SearchKnowledgeArgs]):
    name = "search_knowledge"
    description = "检索校园资料，返回带文档和页码的证据。"
    args_model = SearchKnowledgeArgs

    def __init__(self, retriever: RetrieverLike):
        self.retriever = retriever

    def run(self, arguments: SearchKnowledgeArgs) -> ToolResult:
        query_parts = [arguments.query.strip()]
        if arguments.programme:
            query_parts.append(f"专业：{arguments.programme.strip()}")
        if arguments.admission_year:
            query_parts.append(f"入学年份：{arguments.admission_year}")
        evidence = self.retriever.retrieve("；".join(query_parts))[: arguments.max_results]
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": "；".join(query_parts), "result_count": len(evidence)},
            evidence=[evidence_to_tool_evidence(item) for item in evidence],
        )


class GetCourseTool(BaseTool[GetCourseArgs]):
    name = "get_course"
    description = "按课程代码查询课程目录中的精确课程记录。"
    args_model = GetCourseArgs

    def __init__(self, retriever: RetrieverLike):
        self.retriever = retriever

    def run(self, arguments: GetCourseArgs) -> ToolResult:
        evidence = self.retriever.find_course(arguments.course_code)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"course_code": arguments.course_code, "result_count": len(evidence)},
            evidence=[evidence_to_tool_evidence(item) for item in evidence],
        )


class CompareSchemeVersionsTool(BaseTool[CompareSchemeVersionsArgs]):
    name = "compare_scheme_versions"
    description = "分别检索多个入学年份的培养方案证据，按版本分组返回。"
    args_model = CompareSchemeVersionsArgs

    def __init__(self, retriever: RetrieverLike):
        self.retriever = retriever

    def run(self, arguments: CompareSchemeVersionsArgs) -> ToolResult:
        versions = []
        all_evidence = []
        for year in arguments.admission_years:
            query = arguments.query or "培养方案要求"
            evidence = self.retriever.retrieve(
                f"{arguments.programme} {year}级 {query}"
            )
            scoped = [
                item
                for item in evidence
                if (
                    (match := _APPLICABILITY_PATTERN.match(item.document)) is None
                    or str(year) in match.group("label")
                )
            ]
            start = len(all_evidence)
            all_evidence.extend(scoped)
            versions.append(
                {
                    "admission_year": year,
                    "documents": sorted({item.document for item in scoped}),
                    "evidence_indexes": list(range(start, len(all_evidence))),
                }
            )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"programme": arguments.programme, "versions": versions},
            evidence=[evidence_to_tool_evidence(item) for item in all_evidence],
        )


def build_knowledge_tool_registry(retriever: CampusRetriever) -> ToolRegistry:
    return ToolRegistry(
        [
            SearchKnowledgeTool(retriever),
            GetCourseTool(retriever),
            CompareSchemeVersionsTool(retriever),
        ]
    )
