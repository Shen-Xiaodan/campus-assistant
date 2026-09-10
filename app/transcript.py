"""Transcript PDF parsing and deterministic graduation requirement checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.documents import normalize_text


class TranscriptCourse(BaseModel):
    course_code: str
    course_name: str | None = None
    credits: float | None = Field(default=None, ge=0)
    grade: str | None = None
    term: str | None = None
    status: str = "unknown"
    source_page: int = Field(ge=1)
    confidence: float = Field(ge=0, le=1)


class TranscriptRecord(BaseModel):
    programme: str | None = None
    admission_year: int | None = None
    courses: list[TranscriptCourse]
    total_credits_reported: float | None = None
    warnings: list[str] = Field(default_factory=list)


_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,4})\s*[- ]?\s*(\d{4}[A-Za-z]?)(?![A-Za-z0-9])")
_CREDITS = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:学分|credits?|units?)\b", re.I)
_YEAR = re.compile(r"20\d{2}")


def _status(grade: str | None) -> str:
    if not grade:
        return "unknown"
    value = grade.strip().upper()
    if value in {"W", "WD", "WITHDRAW", "WITHDRAWN"}:
        return "withdrawn"
    if value in {"IP", "IN PROGRESS", "正在修读"}:
        return "in_progress"
    if value in {"F", "FA", "U", "不及格", "FAIL"}:
        return "failed"
    if value in {"A", "A-", "A+", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+", "P", "PASS", "及格"}:
        return "passed"
    return "unknown"


def parse_transcript_pdf(path: Path) -> TranscriptRecord:
    """Parse common text-based transcript layouts; never adds the PDF to RAG."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 pypdf，无法解析成绩单") from exc
    reader = PdfReader(str(path))
    courses: list[TranscriptCourse] = []
    warnings: list[str] = []
    programme = None
    admission_year = None
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            warnings.append(f"第 {page_number} 页没有文本，可能需要 OCR")
            continue
        if programme is None:
            match = re.search(r"(?:Programme|专业|课程|Major)\s*(?:Title|名称)?\s*[:：]?\s*([^\n]+)", text, re.I)
            if match:
                programme = match.group(1).strip()
        if admission_year is None:
            match = re.search(r"(?:入学年份|入学年度|admission year|entry year)\s*[:：]?\s*(20\d{2})", text, re.I)
            if match:
                admission_year = int(match.group(1))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            code_match = _CODE.search(line)
            if not code_match:
                continue
            code = f"{code_match.group(1)}{code_match.group(2)}".upper()
            tail = line[code_match.end() :].strip(" -:|\t")
            if not tail and index + 1 < len(lines):
                tail = lines[index + 1]
            credit_match = _CREDITS.search(line) or (_CREDITS.search(tail) if tail else None)
            credits = float(credit_match.group(1)) if credit_match else None
            grade_match = re.search(r"(?:grade|成绩|result)\s*[:：]?\s*([A-F][+-]?|P|F|IP|W|及格|不及格)", line, re.I)
            grade = grade_match.group(1) if grade_match else None
            courses.append(
                TranscriptCourse(
                    course_code=code,
                    course_name=tail or None,
                    credits=credits,
                    grade=grade,
                    term=None,
                    status=_status(grade),
                    source_page=page_number,
                    confidence=0.75 if credits is not None else 0.55,
                )
            )
    if not courses:
        warnings.append("未识别到课程记录，请确认 PDF 是电子成绩单或改用 OCR")
    return TranscriptRecord(programme=programme, admission_year=admission_year, courses=courses, warnings=warnings)


def normalize_code(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def check_graduation(transcript: TranscriptRecord, requirements: dict[str, Any]) -> dict[str, Any]:
    passed = {
        normalize_code(c.course_code): c
        for c in transcript.courses
        if c.status in {"passed", "exempted", "transferred"}
    }
    in_progress = {normalize_code(c.course_code) for c in transcript.courses if c.status == "in_progress"}
    groups = []
    missing: list[str] = []
    earned = sum(c.credits or 0 for c in passed.values())
    for group in requirements.get("requirement_groups", []):
        required = [normalize_code(code) for code in group.get("courses", [])]
        completed = [code for code in required if code in passed]
        waiting = [code for code in required if code in in_progress]
        missing_codes = [code for code in required if code not in passed and code not in in_progress]
        group_type = group.get("type", "all_of")
        if group_type == "all_of":
            missing.extend(missing_codes)
        elif group_type == "one_of" and not completed:
            missing.extend(missing_codes)
        elif group_type == "choose_n":
            minimum = int(group.get("minimum_courses", 0))
            if len(completed) < minimum:
                missing.extend(missing_codes[: minimum - len(completed)])
        groups.append({
            "id": group.get("id"), "name": group.get("name", group.get("id")),
            "required_credits": group.get("required_credits"), "completed_courses": completed,
            "in_progress_courses": waiting, "missing_courses": missing_codes,
            "completed_count": len(completed), "required_count": len(required),
            "minimum_courses": group.get("minimum_courses"),
        })
    required_credits = float(requirements.get("minimum_total_credits", 0))
    remaining = max(0.0, required_credits - earned)
    status = "satisfied_for_computable_rules" if not missing and remaining == 0 else "partially_satisfied"
    return {
        "overall_status": status, "credits": {"earned": earned, "required": required_credits, "remaining": remaining},
        "requirement_groups": groups, "missing_required_courses": sorted(set(missing)),
        "manual_review_items": requirements.get("manual_review_rules", []),
        "warnings": transcript.warnings,
    }
