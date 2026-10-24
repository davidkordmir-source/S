"""File-backed scene readers. Only previews, visible tiles and model crops enter RAM."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .ecw_reader import check_scene_index_budget, is_ecw
from .errors import UserError
from .image_io import PIL_FORMATS, read_picture

MAX_OUTPUT_EDGE = 4096
PIL_DECODE_PIXELS = 64_000_000
MAX_SINGLE_READ = 64 * 1024 * 1024


def output_size(width: int, height: int, maximum: int) -> tuple[int, int]:
    ratio = min(1, maximum / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


class LimitedFile:
    """Let streamable Pillow decoders seek anywhere without an unbounded read()."""

    def __init__(self, file):
        self.file = file

    def read(self, size=-1):
        if size < 0:
            remaining = os.fstat(self.file.fileno()).st_size - self.file.tell()
            if remaining > MAX_SINGLE_READ:
                raise UserError(
                    "این کدک به خواندن کامل فایل در حافظه نیاز دارد؛ برای تصاویر بسیار بزرگ از ECW یا GeoTIFF استفاده کن."
                )
        elif size > MAX_SINGLE_READ:
            raise UserError("بخش موردنیاز این کدک بیش از بودجهٔ حافظه است؛ تصویر را به GeoTIFF تبدیل کن.")
        return self.file.read(size)

    def __getattr__(self, name):
        return getattr(self.file, name)


class SceneReader:
    def __init__(self, path: Path, metadata: dict | None = None):
        self.path = Path(path)
        self.picture = None
        self.dataset = None
        self.ecw = None
        self.environment = None
        self.file = None
        self.notes = []
        self.meta = metadata or {}
        try:
            if not self.path.is_file():
                raise UserError("فایل اصلی پیدا نشد. اگر آن را جابه‌جا کرده‌ای، دوباره از مسیر جدید بازش کن.")
            with self.path.open("rb") as file:
                header = file.read(32)
            kind = self.meta.get("kind")
            if is_ecw(header):
                from ecw2tiff import EcwReader

                os.environ.setdefault("RAYON_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
                check_scene_index_budget(self.path)
                self.ecw = EcwReader(str(self.path))
                info = self.ecw.info
                self.width, self.height = info["width"], info["height"]
                self.meta = {"kind": "ecw", "version": info["version"], "levels": info["levels"]}
            else:
                opened = False
                # Preserve the established colour/EXIF conversion for small images.
                if kind != "rasterio" and self.path.stat().st_size <= MAX_SINGLE_READ:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", Image.DecompressionBombWarning)
                            with Image.open(self.path) as probe:
                                small = (
                                    probe.width * probe.height <= 16_000_000 and probe.format in PIL_FORMATS
                                )
                        if small:
                            self.picture = read_picture(self.path.read_bytes(), scene=True)
                            self.width, self.height = self.picture.size
                            self.notes = self.picture.info.get("atlas_import_notes", [])
                            self.meta = {"kind": "pillow"}
                            opened = True
                    except (
                        OSError,
                        ValueError,
                        Image.DecompressionBombWarning,
                        Image.DecompressionBombError,
                    ):
                        pass
                if not opened:
                    try:
                        self._open_raster()
                    except UserError:
                        raise
                    except Exception:
                        self._close_raster()
                        self._open_pillow_stream()
            if min(self.width, self.height) < 1 or max(self.width, self.height) > 2**31 - 1:
                raise UserError("ابعاد تصویر معتبر نیست.")
        except (UserError, KeyboardInterrupt, SystemExit):
            self.close()
            raise
        except BaseException as exc:
            self.close()
            raise UserError(
                "فایل تصویر باز نشد؛ فرمت، سلامت فایل و دسترسی به مسیر اصلی را بررسی کن."
            ) from exc

    def _open_pillow_stream(self):
        from .image_io import normalize_open_picture

        self.file = self.path.open("rb")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(LimitedFile(self.file))
        if source.format not in PIL_FORMATS:
            source.close()
            raise UserError("این فرمت در فهرست خواننده‌های برنامه نیست.")
        if source.width * source.height > PIL_DECODE_PIXELS:
            source.close()
            raise UserError(
                "خوانندهٔ این قالب امکان خواندن ناحیه‌ای ندارد؛ برای چنین تصویر بزرگی از ECW یا GeoTIFF استفاده کن."
            )
        try:
            self.picture = normalize_open_picture(source)
        finally:
            source.close()
        self.width, self.height = self.picture.size
        self.notes = self.picture.info.get("atlas_import_notes", [])
        self.meta = {"kind": "pillow"}

    def _open_raster(self):
        import rasterio
        from rasterio.enums import ColorInterp, Resampling

        # GDAL's block cache is process-wide. Keep the bound outside Env so a
        # concurrent reader closing its Env cannot restore a larger cache.
        rasterio.env.set_gdal_config("GDAL_CACHEMAX", 64 * 1024 * 1024)
        self.environment = rasterio.Env(
            GDAL_PAM_ENABLED=False,
            PROJ_NETWORK="OFF",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_VRT_ENABLE_RAWRASTERBAND="NO",
        )
        self.environment.__enter__()
        with self.path.open("rb") as file:
            header = file.read(32)
        driver = (
            "GTiff"
            if header[:4] in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
            else "JPEG"
            if header[:2] == b"\xff\xd8"
            else "PNG"
            if header.startswith(b"\x89PNG")
            else "JP2OpenJPEG"
            if header[:2] == b"\xffO" or header[4:8] == b"jP  "
            else "HFA"
            if header.startswith(b"EHFA_HEADER_TAG")
            else "NITF"
            if header[:4] in (b"NITF", b"NSIF")
            else "PCIDSK"
            if header.startswith(b"PCIDSK")
            else "BMP"
            if header[:2] == b"BM"
            else "GIF"
            if header.startswith(b"GIF8")
            else "WEBP"
            if header[8:12] == b"WEBP"
            else None
        )
        if driver is None:
            raise ValueError("Unknown raster signature")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            self.dataset = rasterio.open(self.path, driver=driver)
        self.width, self.height = self.dataset.width, self.dataset.height
        if any(
            h * w * np.dtype(dtype).itemsize > MAX_SINGLE_READ
            for (h, w), dtype in zip(self.dataset.block_shapes, self.dataset.dtypes, strict=True)
        ):
            raise UserError(
                "بلوک داخلی فایل برای حافظهٔ محدود بیش از حد بزرگ است؛ یک GeoTIFF کاشی‌بندی‌شده بساز."
            )
        colors = self.dataset.colorinterp
        if all(color in colors for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
            self.bands = [
                colors.index(color) + 1 for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            ]
        else:
            self.bands = [1, 2, 3] if self.dataset.count >= 3 else [1]
        self.palette = self.dataset.colormap(1) if colors[0] == ColorInterp.palette else None
        self.alpha = colors.index(ColorInterp.alpha) + 1 if ColorInterp.alpha in colors else None
        self.overviews = self.dataset.overviews(1)
        self.ranges = self.meta.get("ranges")
        if self.ranges is None:
            self.ranges = []
            if any(self.dataset.dtypes[band - 1] != "uint8" for band in self.bands):
                w, h = output_size(self.width, self.height, 256)
                if self.width * self.height / max([1, *self.overviews]) ** 2 > 64_000_000:
                    sample = self._sample_values()
                else:
                    sample = self.dataset.read(
                        self.bands,
                        out_shape=(len(self.bands), h, w),
                        resampling=Resampling.nearest,
                        masked=True,
                    )
                for array in sample:
                    values = array.compressed()
                    values = values[np.isfinite(values)]
                    lo, hi = np.percentile(values, (0.5, 99.5)) if values.size else (0, 1)
                    self.ranges.append([float(lo), float(hi if hi > lo else lo + 1)])
                self.notes.append("دامنهٔ رستر برای نمایش به RGB هشت‌بیتی تبدیل می‌شود.")
        if self.dataset.count > 4:
            self.notes.append("باندهای رنگی مشخص‌شده یا سه باند اول برای جست‌وجو استفاده می‌شوند.")
        self.overviews = self.dataset.overviews(1)
        self.meta = {
            "kind": "rasterio",
            "ranges": self.ranges,
            "driver": self.dataset.driver,
            "overviews": self.overviews,
        }

    def read(self, box: tuple[int, int, int, int] | None = None, maximum: int = 1024) -> Image.Image:
        box = box or (0, 0, self.width, self.height)
        x0, y0, x1, y1 = box
        if not (
            0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height and 1 <= maximum <= MAX_OUTPUT_EDGE
        ):
            raise UserError("محدودهٔ خواندن تصویر معتبر نیست.")
        width, height = x1 - x0, y1 - y0
        w, h = output_size(width, height, maximum)
        factor = (
            (1 << max(0, self.meta.get("levels", 1) - 1))
            if self.ecw is not None
            else max([1, *getattr(self, "overviews", [])])
        )
        if self.picture is None and width * height / (factor * factor) > 64_000_000:
            return self._coarse(box, maximum)
        if self.picture is not None:
            result = self.picture.crop(box)
            if result.size != (w, h):
                result = result.resize((w, h), Image.Resampling.LANCZOS)
            return result.convert("RGB")
        if self.ecw is not None:
            scale = 1 << max(0, math.ceil(math.log2(max(width, height) / maximum)))
            try:
                raw, rw, rh, channels = self.ecw.tile(x0, y0, width, height, scale=scale, encode="raw")
                if channels not in (1, 3, 4) or len(raw) != rw * rh * channels:
                    raise ValueError("Invalid ECW crop")
                result = Image.frombytes({1: "L", 3: "RGB", 4: "RGBA"}[channels], (rw, rh), raw).convert(
                    "RGB"
                )
                return result.resize((w, h), Image.Resampling.LANCZOS) if result.size != (w, h) else result
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                raise UserError("خواندن این ناحیه از ECW کامل نشد.") from exc
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        array = self.dataset.read(
            self.bands,
            window=Window(x0, y0, width, height),
            out_shape=(len(self.bands), h, w),
            resampling=Resampling.bilinear,
            masked=True,
        )
        if self.palette:
            table = np.zeros((max(self.palette) + 1, 3), dtype=np.uint8)
            for key, value in self.palette.items():
                table[key] = value[:3]
            pixels = table[np.clip(array[0].filled(0), 0, len(table) - 1)]
        else:
            values = array.filled(0)
            if self.ranges:
                values = values.astype(np.float32)
                for i, (lo, hi) in enumerate(self.ranges):
                    values[i] = (values[i] - lo) * (255 / (hi - lo))
            values = np.nan_to_num(values, nan=0, posinf=255, neginf=0)
            pixels = np.clip(values, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            if pixels.shape[2] == 1:
                pixels = np.repeat(pixels, 3, axis=2)
        if self.alpha:
            alpha = self.dataset.read(
                self.alpha,
                window=Window(x0, y0, width, height),
                out_shape=(h, w),
                resampling=Resampling.nearest,
            )
            maximum_alpha = 1 if np.issubdtype(alpha.dtype, np.floating) else np.iinfo(alpha.dtype).max
            alpha = np.clip(alpha.astype(np.float32) / maximum_alpha, 0, 1)
            pixels = np.rint(pixels * alpha[:, :, None] + 255 * (1 - alpha[:, :, None])).astype(np.uint8)
        return Image.fromarray(pixels, "RGB")

    def _sample_values(self):
        from rasterio.windows import Window

        rows = []
        for y in np.linspace(0, self.height - 1, 16, dtype=int):
            row = []
            for x in np.linspace(0, self.width - 1, 16, dtype=int):
                row.append(self.dataset.read(self.bands, window=Window(int(x), int(y), 1, 1))[:, 0, 0])
            rows.append(row)
        return np.ma.asarray(np.asarray(rows).transpose(2, 0, 1))

    def _coarse(self, box, maximum):
        x0, y0, x1, y1 = box
        result = Image.new("RGB", (16, 16))
        for j, y in enumerate(np.linspace(y0, y1 - 1, 16, dtype=int)):
            for i, x in enumerate(np.linspace(x0, x1 - 1, 16, dtype=int)):
                result.putpixel(
                    (i, j), self.read((int(x), int(y), int(x) + 1, int(y) + 1), 1).getpixel((0, 0))
                )
        result = result.resize(output_size(x1 - x0, y1 - y0, maximum), Image.Resampling.NEAREST)
        result.info["atlas_import_notes"] = [
            "فایل هرم تصویری کافی ندارد؛ نمای کلی نمونه‌برداری شده است. برای جزئیات بزرگ‌نمایی کن."
        ]
        return result

    def _close_raster(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
        if self.environment is not None:
            self.environment.__exit__(None, None, None)
            self.environment = None

    def close(self):
        self._close_raster()
        self.ecw = None
        if self.picture is not None:
            self.picture.close()
            self.picture = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
