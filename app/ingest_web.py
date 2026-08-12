"""CLI for crawling and incrementally indexing official campus websites."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import PROJECT_ROOT, Settings
from app.logging_config import configure_logging
from app.web_index import build_web_index


def main() -> int:
    parser = argparse.ArgumentParser(description="采集学校官网正文并增量构建向量索引")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data/web_sources.json")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.validate()
    configure_logging(settings.log_level)
    report = build_web_index(settings, args.config.resolve(), args.timeout)
    print(f"成功采集页面: {report.crawled_pages}")
    print(f"新增/更新页面: {report.indexed_pages}")
    print(f"未变化跳过: {report.skipped_pages}")
    print(f"新增文本块: {report.chunks_added}")
    for warning in report.warnings:
        print(f"提示: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
