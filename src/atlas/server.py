from __future__ import annotations

import hmac
import logging
import secrets
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_CENTER, data_directory
from .engine import Engine
from .errors import UserError
from .imagery import MAX_ARCHIVE_BYTES, MAX_IMAGE_BYTES, PDOK, import_archive, save_image
from .jobs import Job, Jobs
from .map_tiles import MapTiles
from .store import Store

logger = logging.getLogger(__name__)


class Area(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius: int = Field(default=250, ge=100, le=600)
    limit: int = Field(default=30, ge=5, le=100)


class TextSearch(BaseModel):
    text: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=12, ge=1, le=50)
    area: Area | None = None


class Services:
    def __init__(self, root: Path, engine: Engine | None = None, imagery: PDOK | None = None):
        self.root = root
        self.store = Store(root)
        self.engine = engine or Engine(root)
        self.imagery = imagery or PDOK()
        self.map_tiles = MapTiles(root, self.imagery)
        self.jobs = Jobs()

    def index(self, job: Job) -> dict:
        pending = self.store.pending()
        if not pending:
            if not self.store.counts()["total"]:
                raise UserError("ابتدا تصاویر محدوده را دریافت یا یک مجموعه را وارد کنید.")
            return {"indexed": 0, "message": "نمایهٔ همهٔ تصاویر به‌روز است."}
        for number, row in enumerate(pending):
            job.update(
                number / len(pending) * 98,
                f"ساخت نمایهٔ تصویر {number + 1} از {len(pending)}؛ بار اول دانلود مدل لازم است",
            )
            vector = self.engine.image(Path(row["image"]))
            job.check()
            self.store.embed(row["id"], vector)
        return {"indexed": len(pending), "message": "نمایه آماده است. می‌توانید با عکس یا توضیح جست‌وجو کنید."}

    def close(self) -> None:
        self.jobs.close()
        self.imagery.close()


async def read_upload(request: Request, limit: int) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, "فایل از حد مجاز بزرگ‌تر است.")
    if not data:
        raise HTTPException(400, "فایلی ارسال نشده است.")
    return bytes(data)


def ui_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "ui"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2] / "ui/dist"


def create_app(
    root: Path | None = None,
    *,
    token: str | None = None,
    services: Services | None = None,
    ui: Path | None = None,
) -> FastAPI:
    session = token or secrets.token_urlsafe(32)
    services = services or Services(root or data_directory())
    ui = ui or ui_directory()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        services.close()

    app = FastAPI(
        title="Atlas Scout",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.token = session
    app.state.services = services

    @app.middleware("http")
    async def protect_local_api(request: Request, call_next):
        if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Invalid local host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            credential = request.cookies.get("atlas_session", "")
            if not hmac.compare_digest(credential, session):
                return JSONResponse(
                    {"detail": "نشست معتبر نیست؛ برنامه را دوباره باز کنید."}, status_code=401
                )
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin access denied"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob: https://tile.openstreetmap.org https://service.pdok.nl; "
            "connect-src 'self' https://service.pdok.nl; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/map/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(UserError)
    async def user_error(request: Request, exc: UserError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/state")
    def state():
        return {
            "version": __version__,
            "library": services.store.counts(),
            "models": services.engine.status(),
            "center": DEFAULT_CENTER,
            "data_directory": str(services.root),
            "provider": "PDOK · Netherlands",
        }

    @app.get("/api/buildings")
    def buildings():
        return {"buildings": services.store.list()}

    @app.get("/api/images/{id}")
    def image(id: str):
        path = services.store.image_path(id)
        if path is None:
            raise HTTPException(404, "تصویر پیدا نشد.")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/map/{kind}/{z}/{x}/{y}")
    def map_tile(kind: str, z: int, x: int, y: int):
        content, mime, freshness = services.map_tiles.get(kind, z, x, y)
        cache_control = f"private, max-age={freshness}" if freshness else "no-store"
        return Response(content, media_type=mime, headers={"Cache-Control": cache_control})

    @app.post("/api/area")
    def prepare_area(area: Area):
        return services.jobs.submit(
            "area",
            lambda job: services.imagery.prepare(
                services.store, job, area.lat, area.lon, area.radius, area.limit
            ),
        )

    @app.post("/api/import")
    async def import_collection(request: Request):
        data = await read_upload(request, MAX_ARCHIVE_BYTES)
        return services.jobs.submit("import", lambda job: import_archive(data, services.store, job))

    @app.post("/api/index")
    def index():
        return services.jobs.submit("index", services.index)

    @app.post("/api/models")
    def prepare_models():
        def task(job: Job):
            job.update(5, "آماده‌سازی مدل تصویر و مدل چندزبانه؛ اولین دانلود ممکن است چند دقیقه طول بکشد")
            services.engine.prepare()
            return {"message": "هر دو مدل هوش مصنوعی آماده‌اند."}

        return services.jobs.submit("models", task)

    @app.post("/api/search/text")
    def search_text(query: TextSearch):
        text = query.text.strip()
        if len(text) < 2:
            raise UserError("توضیح ساختمان را وارد کنید.")
        if not services.store.counts()["indexed"]:
            raise UserError("ابتدا تصاویر کتابخانه را نمایه‌سازی کنید.")

        def task(job: Job):
            job.update(15, "درک توضیح و مقایسه با تصاویر ساختمان‌ها")
            vector = services.engine.text(text)
            job.check()
            return {
                "buildings": services.store.search(
                    vector, query.limit, query.area.model_dump() if query.area else None
                ),
                "query": text,
                "mode": "text",
            }

        return services.jobs.submit("search", task)

    @app.post("/api/search/image")
    async def search_image(
        request: Request,
        lat: float | None = Query(default=None, ge=-90, le=90),
        lon: float | None = Query(default=None, ge=-180, le=180),
        radius: int | None = Query(default=None, ge=100, le=600),
    ):
        if any(value is not None for value in (lat, lon, radius)) and not all(
            value is not None for value in (lat, lon, radius)
        ):
            raise UserError("مرکز و شعاع محدوده را کامل مشخص کنید.")
        area = {"lat": lat, "lon": lon, "radius": radius} if lat is not None else None
        data = await read_upload(request, MAX_IMAGE_BYTES)
        # Validate before queueing so broken files don't start a model download.
        from .imagery import validated_image

        validated_image(data)
        if not services.store.counts()["indexed"]:
            raise UserError("ابتدا تصاویر کتابخانه را نمایه‌سازی کنید.")

        def task(job: Job):
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", dir=services.root, delete=False) as file:
                    temporary = Path(file.name)
                save_image(data, temporary)
                job.update(20, "استخراج ویژگی‌های تصویر و یافتن ساختمان‌های مشابه")
                vector = services.engine.image(temporary)
                job.check()
                return {"buildings": services.store.search(vector, area=area), "mode": "image"}
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)

        return services.jobs.submit("search", task)

    @app.get("/api/jobs/{id}")
    def get_job(id: str):
        job = services.jobs.get(id)
        if not job:
            raise HTTPException(404, "عملیات پیدا نشد.")
        return job.view()

    @app.post("/api/jobs/{id}/cancel")
    def cancel_job(id: str):
        job = services.jobs.get(id)
        if not job:
            raise HTTPException(404, "عملیات پیدا نشد.")
        job.cancel_event.set()
        return {"message": "درخواست لغو ثبت شد؛ پردازش جاری پس از پایان مرحله متوقف می‌شود."}

    @app.get("/")
    def index_page(request: Request, token: str = ""):
        if token and hmac.compare_digest(token, session):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("atlas_session", session, httponly=True, samesite="strict", path="/")
            return response
        index = ui / "index.html"
        if not index.is_file():
            return JSONResponse(
                {"detail": "UI is not built. Run npm ci and npm run build in ui/."}, status_code=503
            )
        return FileResponse(index)

    if (ui / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=ui / "assets"), name="assets")
    return app
