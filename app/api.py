"""FastAPI application for the campus knowledge assistant."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.exceptions import CampusAssistantError, IndexUnavailableError
from app.generation import AnswerGenerator
from app.index import MANIFEST_NAME, open_vectorstore
from app.logging_config import configure_logging
from app.models import ChatRequest, ChatResponse
from app.retrieval import CampusRetriever
from app.service import QAService

logger = logging.getLogger(__name__)


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

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
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
