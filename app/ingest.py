"""CLI entry point for incremental PDF indexing."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import Settings
from app.index import build_index
from app.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导入校园 PDF 并增量构建向量索引")
    parser.add_argument("--source", type=Path, help="PDF 目录，默认使用 DATA_DIR")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.validate()
    configure_logging(settings.log_level)
    report = build_index(settings, args.source.resolve() if args.source else None)
    print(f"新增/更新文档: {report.indexed_documents}")
    print(f"未变化跳过: {report.skipped_documents}")
    print(f"失败文档: {report.failed_documents}")
    print(f"新增文本块: {report.chunks_added}")
    for warning in report.warnings:
        print(f"提示: {warning}")
    return 1 if report.failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
