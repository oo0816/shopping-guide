"""Local browser entry; model calls still require explicit per-request opt-in."""
import logging
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Path as ApiPath, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.graph_workflow import GraphAssistant

WEB = Path(__file__).resolve().parents[1] / "web"
SessionId = Annotated[str, StringConstraints(pattern=r"^[\w.-]{1,64}$", min_length=1, max_length=64)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: SessionId
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    online: StrictBool = False
    method: Literal["vector", "keyword"] = "vector"


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: SessionId
    online: StrictBool = False


def public_result(result):
    """Only send UI fields, not checkpoints, raw model checks or local source paths."""
    keys = ("query", "status", "session_id", "turn", "paused", "resumable", "review_pending",
            "questions", "unsupported_constraints", "requirements", "message", "error",
            "state_saved", "checkpoint_saved", "relative_price_ceiling")
    output = {key: result[key] for key in keys if key in result}
    output["candidates"] = []
    for card in result.get("candidates", []):
        item = {key: card[key] for key in ("product_id", "title", "brand", "category", "sub_category", "sku") if key in card}
        item["fit_review"] = [{key: check[key] for key in ("requirement", "status", "reason", "conditions") if key in check}
                              for check in card.get("fit_review", [])]
        output["candidates"].append(item)
    answer = result.get("answer") or {}
    output["answer"] = {key: answer[key] for key in ("status", "claims", "missing_information") if key in answer} or None
    return output


def create_app(assistant_factory=None):
    def configured_assistant():
        base_url = os.environ.get("SHOP_CATALOG_URL", "").strip()
        if not base_url:
            return GraphAssistant()
        from app.catalog import DEFAULT_DB
        from app.retrieval import ROOT
        from app.shopping import ShoppingAssistant
        from app.shop_catalog import ShopCatalogProvider
        shopping = ShoppingAssistant(candidate_provider=ShopCatalogProvider(base_url, DEFAULT_DB))
        return GraphAssistant(db_path=ROOT / "data/processed/shopping_shop_checkpoints.sqlite", shopping=shopping)

    factory = assistant_factory or configured_assistant
    app = FastAPI(title="有据选购", version="0.1.0", description="本地导购聊天入口；在线调用须显式允许。")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or parsed.netloc != request.headers.get("host"):
                return JSONResponse({"detail": "请从当前服务的页面访问。"}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path not in ("/docs", "/redoc", "/docs/oauth2-redirect"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'"
        return response

    def execute(session_id, **kwargs):
        try:
            return factory().run(session_id, **kwargs)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except (OSError, sqlite3.Error):
            raise HTTPException(503, "会话存储暂时不可用，请稍后恢复会话查看进度。") from None

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logging.getLogger(__name__).error("请求异常：%s", type(exc).__name__)
        return JSONResponse({"detail": "处理暂未完成，请恢复会话查看进度，避免重复发送。"}, status_code=500)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/chat")
    def chat(body: ChatRequest):
        return public_result(execute(body.session_id, query=body.query, online=body.online, method=body.method))

    @app.post("/api/retry")
    def retry(body: RetryRequest):
        return public_result(execute(body.session_id, retry=True, online=body.online))

    @app.get("/api/sessions/{session_id}")
    def session(session_id: Annotated[str, ApiPath(pattern=r"^[\w.-]{1,64}$", min_length=1, max_length=64)]):
        result = execute(session_id, inspect_only=True, include_history=True)
        output = public_result(result)
        output["history"] = [{"query": turn["query"], "result": public_result(turn["result"])} for turn in result.get("history", [])]
        return output

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app


app = create_app()
