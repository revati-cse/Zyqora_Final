"""FastAPI application. Create with create_app(service); run with serve_zyqora.py."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .schemas import (
    BatchItemError, BatchResponse, Classification, CreateUser, ErrorBody, ErrorInfo, Health, HistoryPage,
    VerifyRequest, VerifyResponse, VoicePlay,
)
from .service import ServiceError, ZyqoraService

log = logging.getLogger("zyqora.api")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg", "application/octet-stream"}
ERRORS = {"model": ErrorBody, "description": "Structured error ({error: {code, message, details}})"}


def _err(code: str, message: str, status: int, details: Optional[dict] = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details or {}}}, status_code=status)


async def _read_upload(service: ZyqoraService, file: UploadFile) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise ServiceError("UNSUPPORTED_MEDIA_TYPE", f"{file.content_type} is not supported; use JPEG, PNG or WebP", 415)
    data = await file.read(service.settings.max_upload_bytes + 1)
    if not data:
        raise ServiceError("EMPTY_FILE", "Uploaded file is empty", 422)
    if len(data) > service.settings.max_upload_bytes:
        raise ServiceError("FILE_TOO_LARGE", f"Image is larger than {service.settings.max_upload_bytes // 1048576} MB", 413)
    return data


def create_app(service: ZyqoraService, *, static_dir: Optional[Path] = None, cors_origins: Optional[list[str]] = None,
               load_model: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model and not service.state.ready:
            service.load_model()
        yield
        service.close()
        log.info("server stopped")

    app = FastAPI(title="Zyqora AI API", version=__version__, lifespan=lifespan,
                  description="Offline waste classification on Raspberry Pi 5 (ExecuTorch + XNNPACK).")
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"], allow_credentials=False)

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return _err(exc.code, exc.message, exc.status, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return _err("VALIDATION_ERROR", "Invalid request", 422, {"errors": json.loads(json.dumps(exc.errors(), default=str))})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return _err("INTERNAL_ERROR", "Internal server error (see server log)", 500)

    # ---------------------------------------------------------------- health / model
    @app.get("/api/health", response_model=Health)
    def health():
        dev = service.telemetry.snapshot_device()
        return Health(status="ok", version=__version__, model_ready=service.state.ready, model_code=service.state.code,
                      profile=service.state.profile, device_model=dev["device_model"], machine=dev["machine"],
                      is_raspberry_pi=dev["is_raspberry_pi"], started_at=service.started_at)

    @app.get("/api/model/status")
    def model_status():
        return service.model_status()

    @app.post("/api/model/reload")
    def model_reload():
        service.load_model()
        return service.model_status()

    @app.get("/api/model/metrics")
    def model_metrics():
        return service.model_metrics()

    # ---------------------------------------------------------------- inference
    @app.post("/api/inference", response_model=Classification, responses={413: ERRORS, 415: ERRORS, 422: ERRORS, 503: ERRORS})
    async def inference(file: UploadFile = File(...), user_id: Optional[str] = Form(None),
                        batch_id: Optional[str] = Form(None), ground_truth: Optional[str] = Form(None),
                        source: str = Form("upload")):
        data = await _read_upload(service, file)
        src = source if source in ("upload", "browser-camera", "batch") else "upload"
        if batch_id is not None and not (1 <= len(batch_id) <= 64 and batch_id.replace("-", "").isalnum()):
            raise ServiceError("INVALID_BATCH_ID", "batch_id must be 1-64 letters, digits or '-'", 422)
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(service.classify_bytes, data, file.filename or "upload", source=src,
                                       user_id=user_id or None, batch_id=batch_id, ground_truth=ground_truth or None)

    @app.post("/api/inference/batch", response_model=BatchResponse, responses={503: ERRORS})
    async def inference_batch(files: list[UploadFile] = File(...), ground_truths: Optional[str] = Form(None),
                              user_id: Optional[str] = Form(None)):
        from starlette.concurrency import run_in_threadpool

        service.require_model()
        if len(files) > service.settings.max_batch_files:
            raise ServiceError("TOO_MANY_FILES", f"at most {service.settings.max_batch_files} files per batch", 413)
        try:
            truths = json.loads(ground_truths) if ground_truths else {}
            if not isinstance(truths, dict):
                raise ValueError
        except ValueError as exc:
            raise ServiceError("VALIDATION_ERROR", "ground_truths must be a JSON object {filename: category}", 422) from exc
        batch_id = uuid.uuid4().hex[:12]
        results, errors = [], []
        for f in files:
            name = f.filename or "upload"
            try:
                data = await _read_upload(service, f)
                results.append(await run_in_threadpool(service.classify_bytes, data, name, source="batch",
                                                       user_id=user_id or None, batch_id=batch_id,
                                                       ground_truth=truths.get(name) or None))
            except ServiceError as exc:
                if exc.code in ("SERVER_BUSY",) or exc.status == 503:
                    raise
                errors.append(BatchItemError(filename=name, error=ErrorInfo(code=exc.code, message=exc.message)))
        return BatchResponse(batch_id=batch_id, results=results, errors=errors,
                             summary=service.batch_summary(batch_id)["summary"])

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: str):
        return service.batch_summary(batch_id)

    # ---------------------------------------------------------------- camera
    @app.get("/api/camera/status")
    def camera_status():
        return service.camera_status()

    @app.post("/api/camera/capture", response_model=Classification, responses={503: ERRORS})
    def camera_capture(user_id: Optional[str] = Form(None)):
        return service.camera_capture(user_id=user_id or None)

    # ---------------------------------------------------------------- history / analytics
    @app.get("/api/history", response_model=HistoryPage)
    def history(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                date_from: Optional[str] = None, date_to: Optional[str] = None, category: Optional[str] = None,
                status: Optional[str] = Query(None, pattern="^(ACCEPT|REVIEW)$"), batch_id: Optional[str] = None,
                source: Optional[str] = None):
        items, total = service.store.list(limit=limit, offset=offset, date_from=date_from, date_to=date_to,
                                          category=category, status=status, batch_id=batch_id, source=source)
        for it in items:
            it["image_url"] = f"/api/history/{it['id']}/image" if it["has_thumbnail"] else None
        return HistoryPage(items=items, total=total, limit=limit, offset=offset)

    @app.get("/api/history/export.csv")
    def history_csv(date_from: Optional[str] = None, date_to: Optional[str] = None, category: Optional[str] = None,
                    status: Optional[str] = Query(None, pattern="^(ACCEPT|REVIEW)$")):
        csv_text = service.store.export_csv(date_from=date_from, date_to=date_to, category=category, status=status)
        return PlainTextResponse(csv_text, media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="zyqora_history.csv"'})

    @app.get("/api/history/{item_id}")
    def history_item(item_id: int):
        it = service.store.get(item_id)
        if not it:
            raise ServiceError("NOT_FOUND", f"no classification {item_id}", 404)
        it["image_url"] = f"/api/history/{item_id}/image" if it["has_thumbnail"] else None
        return it

    @app.get("/api/history/{item_id}/image")
    def history_image(item_id: int):
        p = service.thumbnail_path(item_id)
        if p is None:
            raise ServiceError("NOT_FOUND", "no stored image for this classification", 404)
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/analytics/summary")
    def analytics(date_from: Optional[str] = None, date_to: Optional[str] = None, category: Optional[str] = None):
        cats = service.state.loaded.category_set.category_names if service.state.loaded else []
        return service.store.summary(cats, date_from=date_from, date_to=date_to, category=category)

    # ---------------------------------------------------------------- device
    @app.get("/api/device/metrics")
    def device_metrics():
        snap = service.telemetry.snapshot()
        snap.update({"model_ready": service.state.ready, "model_profile": service.state.profile,
                     "runtime": "ExecuTorch" if service.state.ready else None,
                     "backend": ("XNNPACK" if service.state.loaded and
                                 "XnnpackBackend" in service.state.loaded.classifier.registered_backends else None),
                     "threads": service.state.loaded.classifier.thread_count if service.state.loaded else None,
                     "last_inference_ms": service.last_inference_ms, "camera": service.camera_status(),
                     "sorter": {"name": service.sorter.name, "hardware": bool(service.sorter.hardware)}})
        return snap

    # ---------------------------------------------------------------- greenscore
    @app.get("/api/greenscore")
    def greenscore_overview():
        gs = service.greenscore()
        return {"users": [dict(u) for u in gs.list_users()], "points_verified_correct": gs.points_correct,
                "points_verified_incorrect": gs.points_incorrect, "bins": sorted(gs.valid_actions or []),
                "rule": "Points are awarded only after a supervisor verifies the bin the item was placed in."}

    @app.post("/api/greenscore/users")
    def greenscore_add_user(body: CreateUser):
        from ..greenscore import GreenScoreError

        try:
            service.greenscore().add_user(body.user_id, body.display_name)
        except GreenScoreError as exc:
            raise ServiceError("GREENSCORE_ERROR", str(exc), 409) from exc
        return {"user_id": body.user_id}

    @app.get("/api/greenscore/{user_id}")
    def greenscore_user(user_id: str):
        from ..greenscore import GreenScoreError

        gs = service.greenscore()
        try:
            points = gs.points(user_id)
            events = [dict(e) for e in gs.list_events(user_id=user_id)]
        except GreenScoreError as exc:
            raise ServiceError("NOT_FOUND", str(exc), 404) from exc
        contribution: dict[str, int] = {}
        counts = {"PENDING": 0, "VERIFIED_CORRECT": 0, "VERIFIED_INCORRECT": 0}
        for e in events:
            counts[e["verification_status"]] = counts.get(e["verification_status"], 0) + 1
            if e["awarded_points"]:
                key = e["verified_action"] or "?"
                contribution[key] = contribution.get(key, 0) + e["awarded_points"]
        return {"user_id": user_id, "total_points": points, "counts": counts, "contribution_by_bin": contribution,
                "events": list(reversed(events))}

    @app.post("/api/greenscore/verify", response_model=VerifyResponse)
    def greenscore_verify(body: VerifyRequest):
        from ..greenscore import AlreadyVerifiedError, GreenScoreError

        try:
            res = service.greenscore().verify(body.event_id, body.observed_action.upper(),
                                              body.correct_action.upper() if body.correct_action else None,
                                              body.supervisor_id)
        except AlreadyVerifiedError as exc:
            raise ServiceError("ALREADY_VERIFIED", str(exc), 409) from exc
        except GreenScoreError as exc:
            raise ServiceError("GREENSCORE_ERROR", str(exc), 422) from exc
        return VerifyResponse(**res.__dict__)

    # ---------------------------------------------------------------- voice
    @app.get("/api/voice/messages")
    def voice_messages():
        return service.all_messages()

    @app.get("/api/voice/{lang}/{key}.wav")
    def voice_clip(lang: str, key: str):
        p = service.clip(lang, key)
        if p is None:
            raise ServiceError("NOT_FOUND", f"no clip for {lang}/{key}", 404)
        return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/voice/play")
    def voice_play(body: VoicePlay):
        return service.play_on_device(body.lang, body.key)  # sync route: FastAPI runs it in a worker thread

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_404(rest: str):
        raise ServiceError("NOT_FOUND", f"unknown API route /api/{rest}", 404)

    # ---------------------------------------------------------------- frontend (production build)
    if static_dir and (static_dir / "index.html").is_file():
        index = static_dir / "index.html"
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                return FileResponse(candidate)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
