from __future__ import annotations

import csv
import hmac
import io
import json
import secrets
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .config import data_directory
from .engine import Engine
from .errors import UserError
from .image_io import format_support
from .jobs import Jobs
from .workshop import Workshop

MAX_TRAINING_ARCHIVE_BYTES = 150 * 1024 * 1024


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


class ReferenceBox(BaseModel):
    scene: str
    bbox: Rect
    group: str | None = None
    label: Literal[0, 1] = 1
    partition: Literal["train", "test"] = "train"


class ReferenceMember(BaseModel):
    reference: str


class ReferenceEnabled(BaseModel):
    enabled: bool | None = None
    label: Literal[0, 1] | None = None
    partition: Literal["train", "test"] | None = None


class MaskPoint(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    label: Literal[0, 1] = 1


class MaskRequest(BaseModel):
    points: list[MaskPoint] = Field(default_factory=list, max_length=64)
    box: Rect | None = None
    polygon: list[MaskPoint] = Field(default_factory=list, max_length=256)


class ApplyMask(BaseModel):
    reference: str
    draft: str


class IndexRequest(BaseModel):
    scene: str
    block: Literal[256, 512, 1024] = 512
    budget_gb: float = Field(default=20, ge=1, le=4096, allow_inf_nan=False)


class ComputeSettings(BaseModel):
    device: Literal["auto", "cpu", "gpu"] = "auto"
    performance: Literal["balanced", "maximum"] = "maximum"


class TeachingVerdict(BaseModel):
    label: Literal[0, 1]


class SimilarSettings(BaseModel):
    scene: str
    reference: str | None = None
    group: str | None = None
    width: int = Field(default=128, ge=8, le=8192)
    height: int = Field(default=128, ge=8, le=8192)
    threshold: float = Field(default=0.55, ge=0.4, le=0.99, allow_inf_nan=False)
    scope: Rect | None = None
    broad: bool = False
    rotate: bool = True
    intelligent: bool = True
    aerial: bool = True
    geometry: bool = True
    strict_geometry: bool = False
    fine_pose: bool = True
    adapt: bool = True
    mode: Literal["complete", "fast"] = "complete"
    index_block: Literal[256, 512, 1024] = 512
    fast_regions: int = Field(default=1000, ge=10, le=100000)


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
    app.state.desktop_channel_script = b""

    @app.get("/desktop-channel.js")
    def desktop_channel():
        return Response(app.state.desktop_channel_script, media_type="application/javascript")

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
            "desktop_picker": bool(app.state.desktop_channel_script),
            "image_formats": format_support(),
            "models": services.engine.status(),
            "scenes": services.workshop.scenes(),
            "concepts": services.workshop.concepts(),
            "data_directory": str(services.root),
        }

    @app.post("/api/workspace/scenes")
    async def add_scene(request: Request, name: str = "satellite-image"):
        services.idle()
        path = services.workshop.images / f"{uuid.uuid4().hex}.upload"
        submitted = False
        try:
            size, checked = 0, 0
            with path.open("xb") as output:
                async for chunk in request.stream():
                    if size >= checked:
                        if shutil.disk_usage(path.parent).free < len(chunk) + 256 * 1024 * 1024:
                            raise UserError(
                                "فضای کافی برای ذخیرهٔ فایل نیست؛ در نسخهٔ ویندوز فایل را مستقیم باز کن."
                            )
                        checked = size + 32 * 1024 * 1024
                    output.write(chunk)
                    size += len(chunk)
            if not size:
                raise UserError("فایلی ارسال نشده است.")
            result = services.jobs.submit(
                "import-scene", lambda job: services.workshop.import_scene_file(path, name, job, owned=True)
            )
            submitted = True
            return result
        finally:
            if not submitted:
                path.unlink(missing_ok=True)

    @app.get("/api/workspace/scenes/{id}/preview")
    def preview(id: str):
        services.workshop.scene(id)
        return FileResponse(services.workshop.images / f"{id}-preview.jpg", media_type="image/jpeg")

    @app.get("/api/workspace/scenes/{id}/full")
    def full_image(id: str):
        return preview(id)

    @app.get("/api/workspace/scenes/{id}/tiles/{level}/{x}/{y}")
    def scene_tile(id: str, level: int, x: int, y: int):
        return FileResponse(services.workshop.tile(id, level, x, y), media_type="image/jpeg")

    @app.get("/api/workspace/references")
    def references():
        return {"references": services.workshop.similarity.references()}

    @app.post("/api/workspace/references")
    async def reference_file(
        request: Request,
        name: str = "screenshot",
        group: str | None = None,
        label: int = Query(default=1, ge=0, le=1),
        partition: Literal["train", "test"] = "train",
    ):
        services.idle()
        similar = services.workshop.similarity
        if group:
            similar.library.group(group)
        ref = similar.add_file(await upload(request, 15 * 1024 * 1024), name)
        return similar.library.attach_upload(ref, group, label, partition)

    @app.post("/api/workspace/references/box")
    def reference_box(body: ReferenceBox):
        services.idle()
        similar = services.workshop.similarity
        if body.group:
            similar.library.group(body.group)
        return similar.library.attach_upload(
            similar.add_box(body.scene, body.bbox.model_dump()), body.group, body.label, body.partition
        )

    @app.get("/api/workspace/reference-groups")
    def reference_groups():
        return {
            "groups": services.workshop.similarity.library.groups(),
            "active": services.workshop.similarity.library.active(),
        }

    @app.post("/api/workspace/reference-groups/{id}/select")
    def select_reference_group(id: str):
        services.idle()
        return services.workshop.similarity.library.select(id)

    @app.post("/api/workspace/reference-groups")
    def create_reference_group(body: ConceptName):
        services.idle()
        return services.workshop.similarity.library.create(body.name)

    @app.get("/api/workspace/reference-groups/{id}")
    def reference_group(id: str):
        return services.workshop.similarity.library.group(id)

    @app.patch("/api/workspace/reference-groups/{id}")
    def rename_reference_group(id: str, body: ConceptName):
        services.idle()
        return services.workshop.similarity.library.rename(id, body.name)

    @app.post("/api/workspace/reference-groups/{id}/members")
    def attach_reference(id: str, body: ReferenceMember):
        services.idle()
        return services.workshop.similarity.library.add(id, body.reference)

    @app.patch("/api/workspace/reference-groups/{id}/members/{reference}")
    def enable_reference(id: str, reference: str, body: ReferenceEnabled):
        services.idle()
        return services.workshop.similarity.library.change(
            id, reference, body.enabled, label=body.label, partition=body.partition
        )

    @app.delete("/api/workspace/reference-groups/{id}/members/{reference}")
    def detach_reference(id: str, reference: str):
        services.idle()
        return services.workshop.similarity.library.change(id, reference, remove=True)

    @app.get("/api/workspace/references/{id}/image")
    def reference_image(id: str):
        services.workshop.similarity.reference(id)
        return FileResponse(services.workshop.similarity.folder / f"{id}.png", media_type="image/png")

    @app.post("/api/workspace/similar/plan")
    def similar_plan(body: SimilarSettings):
        options = body.model_dump()
        options["scope"] = body.scope.model_dump() if body.scope else None
        return services.workshop.similarity.plan(**options)

    @app.post("/api/workspace/similar/search")
    def similar_search(body: SimilarSettings):
        options = body.model_dump()
        options["scope"] = body.scope.model_dump() if body.scope else None
        return services.jobs.submit(
            "similar", lambda job: services.workshop.similarity.search(job, **options)
        )

    @app.post("/api/workspace/concepts")
    def new_concept(body: ConceptName):
        services.idle()
        return services.workshop.new_concept(body.name)

    @app.post("/api/compute")
    def configure_compute(body: ComputeSettings):
        services.idle()
        return services.engine.configure(body.device, body.performance)

    @app.get("/api/workspace/scenes/{id}/indexes")
    def indexes(id: str):
        return {"indexes": services.workshop.similarity.feature_store.overview(id)}

    @app.post("/api/workspace/scene-index")
    def build_scene_index(body: IndexRequest):
        return services.jobs.submit(
            "scene-index",
            lambda job: services.workshop.similarity.feature_store.build(
                job, body.scene, body.block, body.budget_gb
            ),
        )

    @app.get("/api/workspace/reference-groups/{id}/learning")
    def target_learning_state(id: str):
        result = services.workshop.similarity.learning.status(id)
        result.pop("model", None)
        return result

    @app.post("/api/workspace/reference-groups/{id}/train")
    def train_target(id: str):
        services.workshop.similarity.library.group(id)
        return services.jobs.submit(
            "adapt-target", lambda job: services.workshop.similarity.learning.train(job, id)
        )

    @app.post("/api/workspace/runs/{id}/teach/{index}")
    def teach_result(id: str, index: int, body: TeachingVerdict):
        services.workshop.run(id)
        return services.jobs.submit(
            "teach-feedback",
            lambda job: services.workshop.similarity.learning.feedback(job, id, index, body.label),
        )

    @app.post("/api/workspace/references/{id}/segment")
    def segment_reference(id: str, body: MaskRequest):
        services.workshop.similarity.reference(id)
        return services.jobs.submit(
            "segment-reference",
            lambda job: services.workshop.similarity.segmentation.propose(
                job,
                id,
                [p.model_dump() for p in body.points],
                body.box.model_dump() if body.box else None,
                [p.model_dump() for p in body.polygon],
            ),
        )

    @app.get("/api/workspace/masks/{id}")
    def mask_image(id: str):
        services.workshop.similarity.segmentation.draft(id)
        return FileResponse(
            services.workshop.similarity.segmentation.folder / f"{id}.png", media_type="image/png"
        )

    @app.get("/api/workspace/masks/{id}/overlay")
    def mask_overlay(id: str):
        services.workshop.similarity.segmentation.draft(id)
        return FileResponse(
            services.workshop.similarity.segmentation.folder / f"{id}-overlay.png", media_type="image/png"
        )

    @app.post("/api/workspace/reference-groups/{id}/apply-mask")
    def apply_reference_mask(id: str, body: ApplyMask):
        services.idle()
        return services.workshop.similarity.segmentation.apply(id, body.reference, body.draft)

    @app.post("/api/workspace/reference-groups/{id}/reset-mask")
    def reset_reference_mask(id: str, body: ReferenceMember):
        services.idle()
        similar = services.workshop.similarity
        source = similar.segmentation.source(body.reference)
        similar.library.replace(id, body.reference, source["id"])
        return similar.library.group(id)

    @app.get("/api/workspace/concepts/{id}/examples")
    def examples(id: str):
        return {"examples": services.workshop.examples(id)}

    @app.post("/api/workspace/annotations")
    def annotate(body: Annotation):
        services.idle()
        return services.workshop.add_box(body.concept, body.scene, body.bbox.model_dump(), body.label)

    @app.post("/api/workspace/concepts/{id}/examples")
    async def add_example(id: str, request: Request, label: int = Query(default=1, ge=0, le=1)):
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
    def run(
        id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
        level: Literal["all", "high", "medium", "review"] = "all",
        focus: int | None = Query(None, ge=0),
    ):
        if focus is not None:
            services.workshop.run(id)
            offset = services.workshop.similarity.page_for(id, focus, level, limit)
        return services.workshop.run(id, offset, limit, level)

    @app.post("/api/workspace/runs/{id}/visible")
    def visible_detections(id: str, body: Rect, level: Literal["all", "high", "medium", "review"] = "all"):
        run = services.workshop.run(id)
        if run["settings"].get("method") != "similar":
            return {"detections": run["detections"], "truncated": False}
        return services.workshop.similarity.results(id, limit=2000, viewport=body.model_dump(), level=level)

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
        detection = services.workshop.detection(id, index)
        return services.workshop.add_box(body.concept, run["scene"], detection["bbox"], body.label)

    @app.get("/api/workspace/runs/{id}/csv")
    def export_csv(id: str):
        run = services.workshop.run(id)

        def rows():
            output = io.StringIO()
            writer = csv.writer(output)
            output.write("\ufeff")
            similar = run["settings"].get("method") == "similar"
            writer.writerow(
                ["index", "x", "y", "width", "height", "model_score"]
                + (
                    [
                        "similarity_percent",
                        "level",
                        "pixel_similarity",
                        "structure_similarity",
                        "detail_similarity",
                        "semantic_similarity",
                        "matched_reference",
                        "supporting_views",
                        "aerial_similarity",
                        "geometry_status",
                        "geometry_score",
                        "geometry_inliers",
                        "rotation_degrees",
                        "relative_scale",
                        "target_model_score",
                        "geometry_quad",
                        "search_mode",
                        "searched_tiles",
                        "full_tiles",
                    ]
                    if similar
                    else []
                )
            )
            for entry in services.workshop.iter_detections(run):
                box = entry["bbox"]
                writer.writerow(
                    [entry["index"] + 1, box["x"], box["y"], box["width"], box["height"], entry["score"]]
                    + (
                        [
                            entry["similarity_percent"],
                            entry["level"],
                            entry["appearance"],
                            entry["structure"],
                            entry["detail"],
                            entry["semantic"],
                            entry["matched_reference"],
                            entry["support"],
                            entry.get("aerial"),
                            entry.get("geometry_status"),
                            entry.get("geometry"),
                            entry.get("inliers"),
                            entry.get("rotation"),
                            entry.get("scale"),
                            entry.get("adapted"),
                            json.dumps(entry.get("quad")),
                            run["settings"].get("mode", "complete"),
                            run["settings"].get("windows"),
                            run["settings"].get("full_windows", run["settings"].get("windows")),
                        ]
                        if similar
                        else []
                    )
                )
                if output.tell() >= 65536:
                    yield output.getvalue().encode("utf-8")
                    output.seek(0)
                    output.truncate(0)
            if output.tell():
                yield output.getvalue().encode("utf-8")

        return StreamingResponse(
            rows(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="atlas-detections.csv"'},
        )

    @app.get("/api/workspace/runs/{id}/annotated")
    def export_annotated(id: str):
        run = services.workshop.run(id)
        picture = services.workshop.annotated(run)
        stream = io.BytesIO()
        picture.save(stream, format="PNG", pnginfo=picture.info.get("atlas_png_metadata"))
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
        data = await upload(request, MAX_TRAINING_ARCHIVE_BYTES)
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
