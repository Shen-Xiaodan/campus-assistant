"""FastAPI application for the campus knowledge assistant."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings
from app.exceptions import CampusAssistantError, IndexUnavailableError
from app.generation import AnswerGenerator
from app.index import MANIFEST_NAME, open_vectorstore
from app.logging_config import configure_logging
from app.models import ChatRequest, ChatResponse
from app.retrieval import CampusRetriever
from app.service import QAService
from app.transcript import TranscriptRecord, check_graduation, parse_transcript_pdf

logger = logging.getLogger(__name__)


def _programme_matches(value: str, rules: dict[str, Any]) -> bool:
    normalized = " ".join(value.casefold().split())
    accepted = [rules.get("programme", ""), *rules.get("programme_aliases", [])]
    return normalized in {" ".join(str(option).casefold().split()) for option in accepted if option}


def build_service(settings: Settings) -> QAService:
    manifest = settings.index_dir / MANIFEST_NAME
    if not manifest.exists():
        raise IndexUnavailableError("尚未建立校园资料索引，请先运行 python -m app.ingest")
    try:
        indexed_documents = json.loads(manifest.read_text(encoding="utf-8")).get("documents", {})
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexUnavailableError("索引清单无效，请重新运行 python -m app.ingest") from exc
    if not indexed_documents:
        raise IndexUnavailableError("校园资料索引为空，请先添加 PDF 并运行 python -m app.ingest")
    vectorstore = open_vectorstore(settings)
    return QAService(CampusRetriever(vectorstore, settings), AnswerGenerator(settings))


def create_app(settings: Settings | None = None, service: Any | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.transcripts = {}
        if service is not None:
            app.state.qa_service = service
        else:
            try:
                app.state.qa_service = build_service(settings)
                app.state.startup_error = None
            except CampusAssistantError as exc:
                app.state.qa_service = None
                app.state.startup_error = str(exc)
                logger.warning("qa_service_unavailable: %s", exc)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="基于校园 PDF 资料、返回页码引用的 RAG 问答 API",
        lifespan=lifespan,
    )

    @app.exception_handler(CampusAssistantError)
    async def domain_error_handler(_: Request, exc: CampusAssistantError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        ready = getattr(request.app.state, "qa_service", None) is not None
        return {"status": "ready" if ready else "degraded"}

    @app.post("/transcripts/parse")
    async def parse_transcript(request: Request, upload: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        if upload.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="只支持 PDF 成绩单")
        content = await upload.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024 or not content.startswith(b"%PDF"):
            raise HTTPException(status_code=413, detail="文件必须是 10 MB 以内的有效 PDF")
        analysis_id = uuid.uuid4().hex
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(content)
                temp_path = handle.name
            record = parse_transcript_pdf(Path(temp_path))
            request.app.state.transcripts[analysis_id] = record.model_dump()
            return {"analysis_id": analysis_id, **record.model_dump()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"成绩单解析失败：{exc}") from exc
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @app.post("/graduation/check")
    async def graduation_check(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        analysis_id = str(payload.get("analysis_id", ""))
        raw = request.app.state.transcripts.get(analysis_id)
        if not raw:
            raise HTTPException(status_code=404, detail="成绩单分析已过期，请重新上传")
        programme = payload.get("programme") or raw.get("programme")
        admission_year = payload.get("admission_year") or raw.get("admission_year")
        if not programme or not admission_year:
            raise HTTPException(status_code=422, detail="请确认专业和入学年份")
        rules_path = settings.data_dir / "graduation_requirements.json"
        if not rules_path.exists():
            raise HTTPException(status_code=503, detail="尚未配置毕业要求规则库")
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        year_start = rules.get("admission_year_start", 0)
        year_end = rules.get("admission_year_end", 9999)
        if not _programme_matches(str(programme), rules) or not (year_start <= int(admission_year) <= year_end):
            raise HTTPException(status_code=422, detail="没有找到该专业和入学年份对应的修读计划")
        canonical_programme = rules["programme"]
        report = check_graduation(TranscriptRecord(**raw), rules)
        return {
            "analysis_id": analysis_id,
            "programme": canonical_programme,
            "detected_programme": raw.get("programme"),
            "admission_year": admission_year,
            "scheme": rules,
            **report,
            "disclaimer": "结果仅供选课规划参考，以教务处最终审核为准。",
        }

    @app.delete("/transcripts/{analysis_id}")
    async def delete_transcript(analysis_id: str, request: Request) -> dict[str, str]:
        request.app.state.transcripts.pop(analysis_id, None)
        return {"status": "deleted"}

    @app.post(
        "/chat",
        response_model=ChatResponse,
        response_model_exclude_none=True,
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        qa_service = getattr(request.app.state, "qa_service", None)
        if qa_service is None:
            detail = getattr(request.app.state, "startup_error", "问答服务尚未就绪")
            raise HTTPException(status_code=503, detail=detail)
        started = time.perf_counter()
        try:
            response = qa_service.ask(payload.question, payload.history)
        except CampusAssistantError:
            raise
        except Exception as exc:
            logger.exception("chat_request_failed")
            raise HTTPException(status_code=500, detail="处理问题时发生内部错误") from exc
        logger.info(
            "chat_request_completed duration_ms=%.2f grounded=%s sources=%d",
            (time.perf_counter() - started) * 1000,
            response.grounded,
            len(response.sources),
        )
        return response

    return app


app = create_app()
