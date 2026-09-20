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


_COURSE_CODE = r"[A-Z]{2,4}\s*[- ]?\s*\d{4}[A-Z]?"
_GRADE = r"A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|PA|DI|IP|WD|W|P|F|U"
_COURSE_ROW = re.compile(
    rf"^\s*(?P<code>{_COURSE_CODE})\s+"
    rf"(?P<title>.+?)\s+"
    rf"(?P<credits>\d+(?:\.\d+)?)\s+"
    rf"(?P<grade>{_GRADE})\s+"
    r"(?P<percentage>N/?A|\d+(?:\.\d+)?)\s*$",
    re.I,
)
_COURSE_START = re.compile(rf"^\s*(?P<code>{_COURSE_CODE})\b", re.I)
_TERM = re.compile(r"^\s*(20\d{2}[-–—]\d{2})\s+Term\s+([12])\s*$", re.I)
_PROGRAMME = re.compile(
    r"^\s*(?:Major\s*/\s*Programme|Programme|Major|专业)\s*[:：]\s*(.+?)\s*$",
    re.I | re.M,
)
_ADMISSION_YEAR = re.compile(
    r"^\s*(?:Admitted\s+in|Admission\s+Year|Entry\s+Year|入学年份|入学年度)\s*[:：]\s*"
    r"(?:[A-Za-z]+\s+)?(20\d{2})\s*$",
    re.I | re.M,
)
_TOTAL_UNITS = re.compile(r"(?:Cumulative\s+Units\s+Passed|累计已获学分)\s*=\s*(\d+(?:\.\d+)?)", re.I)


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
    if value in {
        "A", "A-", "A+", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+",
        "P", "PA", "DI", "PASS", "及格",
    }:
        return "passed"
    return "unknown"


def _is_course_header(line: str) -> bool:
    value = line.casefold()
    return all(label in value for label in ("course code", "course title", "units", "grade"))


def _is_course_table_end(line: str) -> bool:
    value = line.strip().casefold()
    return value.startswith(
        (
            "units passed",
            "cumulative units passed",
            "honour(s)/award(s)",
            "summary",
            "remarks",
            "invalid unless",
            "end of transcript",
        )
    ) or value.startswith("*")


def _is_title_continuation(raw_line: str, line: str) -> bool:
    indentation = len(raw_line) - len(raw_line.lstrip())
    return indentation >= 4 and "=" not in line and not _TERM.match(line) and not _is_course_header(line)


def parse_transcript_pages(page_texts: list[str]) -> TranscriptRecord:
    """Parse layout-preserving transcript page text into a normalized record."""
    courses: list[TranscriptCourse] = []
    warnings: list[str] = []
    programme = None
    admission_year = None
    total_credits: list[float] = []
    nonempty_pages = [index for index, text in enumerate(page_texts, start=1) if text.strip()]
    last_nonempty_page = max(nonempty_pages, default=0)
    all_pages_empty = not nonempty_pages

    for page_number, raw_text in enumerate(page_texts, start=1):
        text = raw_text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            if all_pages_empty or page_number <= last_nonempty_page:
                warnings.append(f"第 {page_number} 页没有文本，可能需要 OCR")
            continue
        if programme is None:
            match = _PROGRAMME.search(text)
            if match:
                programme = match.group(1).strip()
        if admission_year is None:
            match = _ADMISSION_YEAR.search(text)
            if match:
                admission_year = int(match.group(1))

        total_credits.extend(float(value) for value in _TOTAL_UNITS.findall(text))
        current_term = None
        inside_course_table = False
        last_course: TranscriptCourse | None = None
        for raw_line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                continue
            term_match = _TERM.match(line)
            if term_match:
                current_term = f"{term_match.group(1).replace('–', '-').replace('—', '-')} Term {term_match.group(2)}"
                inside_course_table = False
                last_course = None
                continue
            if _is_course_header(line):
                inside_course_table = True
                last_course = None
                continue
            if not inside_course_table:
                continue
            if _is_course_table_end(line):
                inside_course_table = False
                last_course = None
                continue

            row_match = _COURSE_ROW.match(line)
            if row_match:
                grade = row_match.group("grade").upper()
                course = TranscriptCourse(
                    course_code=normalize_code(row_match.group("code")),
                    course_name=row_match.group("title").strip(),
                    credits=float(row_match.group("credits")),
                    grade=grade,
                    term=current_term,
                    status=_status(grade),
                    source_page=page_number,
                    confidence=0.98,
                )
                courses.append(course)
                last_course = course
                continue
            incomplete_match = _COURSE_START.match(line)
            if incomplete_match:
                warnings.append(
                    f"第 {page_number} 页课程 {normalize_code(incomplete_match.group('code'))} 字段不完整，已跳过"
                )
                last_course = None
                continue
            if last_course is not None and _is_title_continuation(raw_line, line):
                separator = "" if last_course.course_name and last_course.course_name.endswith(("-", "–", "—")) else " "
                last_course.course_name = f"{last_course.course_name}{separator}{line}".strip()

    if not courses:
        warnings.append("未识别到课程记录，请确认 PDF 是电子成绩单或改用 OCR")
    return TranscriptRecord(
        programme=programme,
        admission_year=admission_year,
        courses=courses,
        total_credits_reported=max(total_credits, default=None),
        warnings=warnings,
    )


def parse_transcript_pdf(path: Path) -> TranscriptRecord:
    """Parse common text-based transcript layouts; never adds the PDF to RAG."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 pypdf，无法解析成绩单") from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text(extraction_mode="layout") or "")
        except (TypeError, ValueError):
            pages.append(normalize_text(page.extract_text() or ""))
    return parse_transcript_pages(pages)


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
