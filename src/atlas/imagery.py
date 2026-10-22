from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from defusedxml import ElementTree
from PIL import Image, ImageOps, UnidentifiedImageError
from pyproj import Transformer

from .config import WFS_URL, WMS_URL
from .errors import UserError
from .jobs import Job
from .store import Store

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_ARCHIVE_EXPANDED = 160 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
TO_RD = Transformer.from_crs(4326, 28992, always_xy=True)
FROM_RD = Transformer.from_crs(28992, 4326, always_xy=True)


def validated_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise UserError("حجم تصویر باید کمتر از ۱۵ مگابایت باشد.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise UserError("ابعاد تصویر بیش از حد بزرگ است؛ از تصویری کوچک‌تر استفاده کنید.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise UserError("فایل انتخاب‌شده یک تصویر معتبر و قابل‌خواندن نیست.") from exc


def save_image(data: bytes, path: Path) -> None:
    image = validated_image(data)
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    pending = path.with_suffix(".tmp")
    image.save(pending, format="JPEG", quality=92)
    pending.replace(path)


def import_archive(data: bytes, store: Store, job: Job) -> dict:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise UserError("حجم فایل ZIP باید کمتر از ۸۰ مگابایت باشد.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UserError("فایل ZIP معتبر نیست.") from exc
    with archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) > 501:
            raise UserError("فایل ZIP دارای نام تکراری یا تعداد فایل بیش از حد مجاز است.")
        if sum(item.file_size for item in infos) > MAX_ARCHIVE_EXPANDED:
            raise UserError("حجم محتوای بازشده بیش از حد مجاز است.")
        if "buildings.json" not in names or archive.getinfo("buildings.json").file_size > 1024 * 1024:
            raise UserError("فایل buildings.json باید در ریشهٔ ZIP قرار داشته باشد و کمتر از یک مگابایت باشد.")
        try:
            document = json.loads(archive.read("buildings.json"))
            entries = document["buildings"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise UserError("ساختار buildings.json معتبر نیست.") from exc
        if not isinstance(entries, list) or not 1 <= len(entries) <= 200:
            raise UserError("مجموعه باید بین ۱ تا ۲۰۰ ساختمان داشته باشد.")
        staged: list[dict] = []
        identifiers: set[str] = set()
        # Validate every entry before touching the library. Never extract ZIP
        # paths to disk, accept external paths, or trust archive filenames.
        for number, entry in enumerate(entries):
            job.update(number / len(entries) * 45, "بررسی مجموعهٔ تصاویر")
            try:
                if not isinstance(entry, dict):
                    raise ValueError()
                filename = entry["image"]
                if not isinstance(filename, str) or "\\" in filename:
                    raise ValueError()
                path = PurePosixPath(filename)
                if path.is_absolute() or ".." in path.parts or ":" in filename:
                    raise ValueError()
                lat, lon = float(entry["lat"]), float(entry["lon"])
                if (
                    not math.isfinite(lat)
                    or not math.isfinite(lon)
                    or not (-90 <= lat <= 90 and -180 <= lon <= 180)
                ):
                    raise ValueError()
                info = archive.getinfo(filename)
                if info.file_size > MAX_IMAGE_BYTES:
                    raise ValueError()
                raw = archive.read(filename)
                validated_image(raw)
                source_id = str(entry.get("id", filename))
                identifier = "local-" + hashlib.sha256(f"{source_id}:{lat}:{lon}".encode()).hexdigest()[:24]
                if identifier in identifiers:
                    raise ValueError()
                identifiers.add(identifier)
                staged.append(
                    {
                        "id": identifier,
                        "title": str(entry.get("title", f"ساختمان {number + 1}"))[:160],
                        "lat": lat,
                        "lon": lon,
                        "source": str(entry.get("source", "مجموعهٔ واردشده توسط کاربر"))[:240],
                        "raw": raw,
                    }
                )
            except (KeyError, ValueError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
                raise UserError(f"اطلاعات یا تصویر ساختمان شمارهٔ {number + 1} معتبر نیست.") from exc
        for number, entry in enumerate(staged):
            job.update(45 + number / len(staged) * 50, "ذخیرهٔ تصاویر در کتابخانه")
            raw = entry.pop("raw")
            destination = store.images / f"{entry['id']}.jpg"
            save_image(raw, destination)
            store.put(**entry, image=destination)
    return {
        "imported": len(staged),
        "center": {"lat": staged[0]["lat"], "lon": staged[0]["lon"]},
        "message": "تصاویر وارد شدند. برای جست‌وجو، نمایهٔ هوش مصنوعی را بسازید.",
    }


def fetch_bytes(client: httpx.Client, url: str, params: dict, maximum: int) -> tuple[bytes, str]:
    try:
        with client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > maximum:
                    raise UserError("پاسخ سرویس از حد مجاز بزرگ‌تر بود. محدودهٔ کوچک‌تری انتخاب کنید.")
            return bytes(content), response.headers.get("content-type", "")
    except httpx.HTTPStatusError as exc:
        raise UserError(
            f"سرویس تصاویر پاسخ موفق نداد (HTTP {exc.response.status_code}). اتصال و دسترسی به service.pdok.nl را بررسی کنید."
        ) from exc
    except httpx.HTTPError as exc:
        raise UserError(
            "اتصال به service.pdok.nl برقرار نشد. دسترسی شبکه را بررسی و دوباره تلاش کنید."
        ) from exc


class PDOK:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(
            timeout=35,
            follow_redirects=True,
            headers={"User-Agent": "AtlasScout/0.1 (https://github.com/davidkordmir-source/S)"},
        )

    def close(self) -> None:
        self.client.close()

    def layer(self) -> str:
        body, _ = fetch_bytes(
            self.client, WMS_URL, {"service": "WMS", "request": "GetCapabilities"}, 5 * 1024 * 1024
        )
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise UserError("فهرست لایه‌های تصویری سرویس قابل خواندن نیست.") from exc
        names = [node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Name"]
        # Only select an advertised current RGB layer, never an invented layer.
        for candidate in ("Actueel_orthoHR", "Actueel_ortho25"):
            if candidate in names:
                return candidate
        raise UserError("لایهٔ تصاویر هوایی فعلی در سرویس پیدا نشد. آدرس سرویس نیاز به بررسی دارد.")

    def prepare(self, store: Store, job: Job, lat: float, lon: float, radius: int, limit: int) -> dict:
        if not (50.7 <= lat <= 53.6 and 3.2 <= lon <= 7.3):
            raise UserError(
                "دریافت آنلاین نسخهٔ اول برای هلند آماده شده است. داخل هلند محدوده‌ای انتخاب کنید؛ مجموعهٔ شخصی می‌تواند متعلق به هر نقطه باشد."
            )
        job.update(2, "بررسی سرویس تصاویر هوایی PDOK")
        layer = self.layer()
        x, y = TO_RD.transform(lon, lat)
        body, _ = fetch_bytes(
            self.client,
            WFS_URL,
            {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": "bag:pand",
                "outputFormat": "application/json",
                "srsName": "EPSG:28992",
                "bbox": f"{x - radius},{y - radius},{x + radius},{y + radius},EPSG:28992",
                # Get enough features to rank nearby footprints rather than trusting
                # the provider's feature order. We disclose the cap in the result.
                "count": 1000,
            },
            12 * 1024 * 1024,
        )
        try:
            document = json.loads(body)
            features = document["features"]
            if not isinstance(features, list):
                raise ValueError()
        except (KeyError, ValueError, TypeError) as exc:
            raise UserError("سرویس ساختمان‌ها دادهٔ مکانی قابل‌استفاده برنگرداند.") from exc
        candidates = []
        for feature in features:
            geometry = feature.get("geometry") or {}
            if geometry.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            points = list(coordinates(geometry.get("coordinates", [])))
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            bounds = min(xs), min(ys), max(xs), max(ys)
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            if (cx - x) ** 2 + (cy - y) ** 2 > radius**2:
                continue
            candidates.append(((cx - x) ** 2 + (cy - y) ** 2, feature, bounds, cx, cy))
        candidates.sort(key=lambda entry: entry[0])
        if not candidates:
            raise UserError("ساختمانی در این محدوده پیدا نشد. مرکز یا شعاع جست‌وجو را تغییر دهید.")
        selected = candidates[:limit]
        downloaded, reused = 0, 0
        for index, (_, feature, bounds, cx, cy) in enumerate(selected):
            job.update(
                10 + index / len(selected) * 85, f"دریافت تصویر ساختمان {index + 1} از {len(selected)}"
            )
            identifier = "pdok-" + hashlib.sha256(str(feature["id"]).encode()).hexdigest()[:24]
            if store.image_path(identifier):
                reused += 1
                continue
            # Use a square crop preserving the roof and its surroundings.
            half = max(12, (bounds[2] - bounds[0]) / 2 + 5, (bounds[3] - bounds[1]) / 2 + 5)
            if half > 180:
                continue
            image, content_type = fetch_bytes(
                self.client,
                WMS_URL,
                {
                    "service": "WMS",
                    "version": "1.1.1",
                    "request": "GetMap",
                    "layers": layer,
                    "styles": "",
                    "srs": "EPSG:28992",
                    "bbox": f"{cx - half},{cy - half},{cx + half},{cy + half}",
                    "width": 256,
                    "height": 256,
                    "format": "image/jpeg",
                },
                MAX_IMAGE_BYTES,
            )
            if not content_type.startswith("image/"):
                raise UserError("سرویس به‌جای تصویر، پیام خطا فرستاد. لایهٔ تصویری را بررسی کنید.")
            destination = store.images / f"{identifier}.jpg"
            save_image(image, destination)
            building_lon, building_lat = FROM_RD.transform(cx, cy)
            props = feature.get("properties", {})
            source_id = props.get("identificatie", feature["id"])
            store.put(
                id=identifier,
                title=f"ساختمان {source_id}",
                lat=building_lat,
                lon=building_lon,
                source="PDOK · BAG / Luchtfoto RGB",
                image=destination,
                geometry=geographic_geometry(feature["geometry"]),
            )
            downloaded += 1
        return {
            "downloaded": downloaded,
            "reused": reused,
            "available": len(candidates),
            "provider_capped": len(features) >= 1000,
            "message": f"{downloaded} تصویر جدید و {reused} تصویر موجود آماده شد. اکنون نمایه را بسازید.",
        }


def coordinates(value: Any):
    if isinstance(value, list) and len(value) >= 2 and isinstance(value[0], (int, float)):
        if math.isfinite(value[0]) and math.isfinite(value[1]):
            yield value[0], value[1]
    elif isinstance(value, list):
        for child in value:
            yield from coordinates(child)


def geographic_geometry(geometry: dict) -> dict:
    def convert(value):
        if value and isinstance(value[0], (int, float)):
            return list(FROM_RD.transform(value[0], value[1]))
        return [convert(child) for child in value]

    return {"type": geometry["type"], "coordinates": convert(geometry["coordinates"])}
