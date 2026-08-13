"""Lightweight Chinese/English adaptation for campus questions and metadata."""

from __future__ import annotations

import re

DEPARTMENT_NAMES: dict[str, tuple[str, str]] = {
    "学生事务处": ("学生事务处", "Office of Student Affairs"),
    "教务处": ("教务处", "Registry"),
    "图书馆": ("图书馆", "Library"),
    "研究生院": ("研究生院", "Graduate School"),
    "职业规划与发展处": ("职业规划与发展处", "Career Planning and Development Office"),
}


def detect_language(text: str) -> str:
    """Return ``zh`` or ``en`` using dominant script, defaulting to Chinese."""
    chinese_count = len(re.findall(r"[\u3400-\u9fff]", text))
    english_count = len(re.findall(r"[A-Za-z]", text))
    # Latin words contain several letters each, so require a clear majority
    # before treating a mixed campus query (for example one with a course or
    # department name) as English.
    return "en" if english_count > chinese_count * 2 else "zh"


def bilingual_department(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    normalized = name.strip()
    for chinese, english in DEPARTMENT_NAMES.values():
        if normalized.casefold() in {chinese.casefold(), english.casefold()}:
            return chinese, english
    return normalized, None
