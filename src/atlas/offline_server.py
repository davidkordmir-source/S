from __future__ import annotations

import csv
import hmac
import io
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from . import __version__
from .config import data_directory
from .engine import Engine
from .errors import UserError
from .jobs import Jobs
from .workshop import MAX_SCENE_BYTES, Workshop


class Rect(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    width: float = Field(ge=8, allow_inf_nan=False)
    height: float = Field(ge=8, allow_inf_nan=False)


class ConceptName(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Annotation(BaseModel):
    concept: str
    scene: str
    bbox: Rect
    label: Literal[0, 1]


class SearchSettings(BaseModel):
    scene: str
    concept: str | None = None
    text: str = Field(default="", max_length=500)
    width: int = Field(default=96, ge=8, le=2048)
    height: int = Field(default=96, ge=8, le=2048)
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    scope: Rect | None = None
    thorough: bool = False


class Feedback(BaseModel):
    concept: str
    label: Literal[0, 1]


class OfflineServices:
    def __init__(self, root: Path, engine: Engine | None = None):
        self.root = root
        self.engine = engine or Engine(root)
        self.workshop = Workshop(root, self.engine)
        self.jobs = Jobs()

    def idle(self):
        with self.jobs.lock:
            if any(job.status in ("queued", "running") for job in self.jobs.jobs.values()):
                raise UserError(
                    "تا پایان عملیات صبر کن یا آن را لغو کن؛ نمونه‌ها در زمان آموزش تغییر نمی‌کنند."
                )

    def close(self):
        self.jobs.close()


async def upload(request: Request, maximum: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > maximum:
            raise HTTPException(413, "فایل از حد مجاز بزرگ‌تر است.")
    if not body:
        raise UserError("فایلی ارسال نشده است.")
    return bytes(body)


def create_app(
    root: Path | None = None, *, token: str | None = None, services: OfflineServices | None = None
) -> FastAPI:
    services = services or OfflineServices(root or data_directory())
    token = token or secrets.token_urlsafe(32)
    ui = (
        Path(sys._MEIPASS) / "ui"
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2] / "ui/dist"
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        services.close()

    app = FastAPI(title="Atlas Offline", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token, app.state.services = token, services

    @app.middleware("http")
    async def local_only(request, call_next):
        if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Invalid local host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            if not hmac.compare_digest(request.cookies.get("atlas_session", ""), token):
                return JSONResponse(
                    {"detail": "نشست برنامه معتبر نیست. برنامه را دوباره باز کن."}, status_code=401
                )
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin access denied"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(UserError)
    async def user_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/state")
    def state():
        return {
            "version": __version__,
            "offline": True,
            "models": services.engine.status(),
            "scenes": services.workshop.scenes(),
            "concepts": services.workshop.concepts(),
            "data_directory": str(services.root),
        }

    @app.post("/api/workspace/scenes")
    async def add_scene(request: Request, name: str = "satellite-image"):
        data = await upload(request, MAX_SCENE_BYTES)
        return services.jobs.submit(
            "import-scene", lambda job: services.workshop.import_scene(data, name, job)
        )

    @app.get("/api/workspace/scenes/{id}/preview")
    def preview(id: str):
        services.workshop.scene(id)
        return FileResponse(services.workshop.images / f"{id}-preview.jpg", media_type="image/jpeg")

    @app.get("/api/workspace/scenes/{id}/full")
    def full_image(id: str):
        return FileResponse(services.workshop.scene_path(id), media_type="image/png")

    @app.post("/api/workspace/concepts")
    def new_concept(body: ConceptName):
        services.idle()
        return services.workshop.new_concept(body.name)

    @app.get("/api/workspace/concepts/{id}/examples")
    def examples(id: str):
        return {"examples": services.workshop.examples(id)}

    @app.post("/api/workspace/annotations")
    def annotate(body: Annotation):
        services.idle()
        return services.workshop.add_box(body.concept, body.scene, body.bbox.model_dump(), body.label)

    @app.post("/api/workspace/concepts/{id}/examples")
    async def add_example(id: str, request: Request, label: Literal[0, 1] = 1):
        services.idle()
        data = await upload(request, 15 * 1024 * 1024)
        return services.workshop.add_file(id, data, label)

    @app.get("/api/workspace/examples/{id}/image")
    def example_image(id: str):
        return FileResponse(services.workshop.example_path(id), media_type="image/png")

    @app.delete("/api/workspace/examples/{id}")
    def delete_example(id: str):
        services.idle()
        services.workshop.delete_example(id)
        return {"message": "نمونه حذف شد؛ آموزش را به‌روزرسانی کن."}

    @app.post("/api/workspace/concepts/{id}/train")
    def train(id: str):
        return services.jobs.submit("train", lambda job: services.workshop.train(id, job))

    @app.post("/api/workspace/plan")
    def plan(body: SearchSettings):
        result = services.workshop.plan(
            body.scene,
            body.width,
            body.height,
            body.scope.model_dump() if body.scope else None,
            body.thorough,
        )
        result.pop("boxes", None)
        return result

    @app.post("/api/workspace/search")
    def search(body: SearchSettings):
        return services.jobs.submit(
            "scan",
            lambda job: services.workshop.search(
                body.scene,
                body.concept,
                body.text,
                body.width,
                body.height,
                body.threshold,
                body.scope.model_dump() if body.scope else None,
                body.thorough,
                job,
            ),
        )

    @app.get("/api/workspace/runs/{id}")
    def run(id: str):
        return services.workshop.run(id)

    @app.get("/api/workspace/runs/{id}/crops/{index}")
    def detection_image(id: str, index: int):
        image = services.workshop.crop_detection(id, index)
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=90)
        return Response(stream.getvalue(), media_type="image/jpeg")

    @app.post("/api/workspace/runs/{id}/feedback/{index}")
    def feedback(id: str, index: int, body: Feedback):
        services.idle()
        run = services.workshop.run(id)
        if not 0 <= index < len(run["detections"]):
            raise UserError("کادر پیدا نشد.")
        return services.workshop.add_box(
            body.concept, run["scene"], run["detections"][index]["bbox"], body.label
        )

    @app.get("/api/workspace/runs/{id}/csv")
    def export_csv(id: str):
        run = services.workshop.run(id)
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["index", "x", "y", "width", "height", "model_score"])
        for entry in run["detections"]:
            box = entry["bbox"]
            writer.writerow(
                [entry["index"] + 1, box["x"], box["y"], box["width"], box["height"], entry["score"]]
            )
        return Response(
            stream.getvalue().encode("utf-8-sig"),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="atlas-detections.csv"'},
        )

    @app.get("/api/workspace/runs/{id}/annotated")
    def export_annotated(id: str):
        run = services.workshop.run(id)
        with Image.open(services.workshop.scene_path(run["scene"])) as original:
            picture = original.copy()
        draw = ImageDraw.Draw(picture)
        width = max(2, round(max(picture.size) / 700))
        for entry in run["detections"]:
            box = entry["bbox"]
            draw.rectangle(
                (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]),
                outline="#89ed97",
                width=width,
            )
            draw.text(
                (box["x"] + 3, box["y"] + 3),
                str(entry["index"] + 1),
                fill="#ffffff",
                stroke_fill="#193825",
                stroke_width=2,
            )
        stream = io.BytesIO()
        picture.save(stream, format="PNG")
        return Response(
            stream.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="atlas-annotated.png"'},
        )

    @app.get("/api/workspace/concepts/{id}/export")
    def export_training(id: str):
        return Response(
            services.workshop.export_training(id),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="atlas-training.zip"'},
        )

    @app.post("/api/workspace/training/import")
    async def import_training(request: Request):
        data = await upload(request, MAX_SCENE_BYTES)
        return services.jobs.submit(
            "import-training", lambda job: services.workshop.import_training(data, job)
        )

    @app.get("/api/jobs/{id}")
    def job_status(id: str):
        job = services.jobs.get(id)
        if not job:
            raise HTTPException(404, "عملیات پیدا نشد.")
        return job.view()

    @app.post("/api/jobs/{id}/cancel")
    def cancel(id: str):
        job = services.jobs.get(id)
        if not job:
            raise HTTPException(404, "عملیات پیدا نشد.")
        job.cancel_event.set()
        return {"message": "درخواست لغو ثبت شد؛ مرحلهٔ جاری به پایان می‌رسد و عملیات متوقف می‌شود."}

    @app.get("/")
    def index(token_value: str = "", token: str = ""):
        supplied = token or token_value
        if supplied and hmac.compare_digest(supplied, app.state.token):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("atlas_session", app.state.token, httponly=True, samesite="strict", path="/")
            return response
        if not (ui / "index.html").is_file():
            return JSONResponse({"detail": "Build the UI first."}, status_code=503)
        return FileResponse(ui / "index.html")

    if (ui / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=ui / "assets"), name="assets")
    return app
