"""Bounded full-image decoding for examples and legacy small-image callers.

Large scenes use scene_source.SceneReader and never enter this byte API.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from . import ecw_reader, raster_reader
from .errors import UserError

register_heif_opener(thumbnails=False)

MAX_DECODE_BYTES = 150 * 1024 * 1024
MAX_PIXELS = 64_000_000
FORMAT_GROUPS = [
    ("PNG / APNG", (".png", ".apng")),
    ("JPEG / JFIF", (".jpg", ".jpeg", ".jpe", ".jfif")),
    ("TIFF / GeoTIFF", (".tif", ".tiff", ".gtif", ".gtiff")),
    ("ECW", (".ecw",)),
    ("JPEG 2000", (".jp2", ".j2k", ".j2c", ".jpc", ".jpf", ".jpx")),
    ("HEIC / HEIF", (".heic", ".heif", ".hif", ".heics", ".heifs")),
    ("AVIF", (".avif", ".avifs")),
    ("WebP", (".webp",)),
    ("GIF", (".gif",)),
    ("BMP / DIB", (".bmp", ".dib")),
    ("PSD (merged image)", (".psd",)),
    ("TGA", (".tga", ".icb", ".vda", ".vst")),
    ("PCX", (".pcx",)),
    ("PPM / PGM / PBM / PNM", (".ppm", ".pgm", ".pbm", ".pnm", ".pfm")),
    ("SGI", (".sgi", ".rgb", ".rgba", ".bw")),
    ("DDS", (".dds",)),
    ("ICO / ICNS", (".ico", ".icns")),
    ("QOI", (".qoi",)),
    ("XBM / XPM", (".xbm", ".xpm")),
    ("ERDAS Imagine / HFA", (".img",)),
    ("NITF", (".ntf", ".nitf")),
    ("PCIDSK", (".pix",)),
]
PIL_FORMATS = {
    "PNG",
    "JPEG",
    "TIFF",
    "JPEG2000",
    "HEIF",
    "AVIF",
    "WEBP",
    "GIF",
    "BMP",
    "DIB",
    "PSD",
    "TGA",
    "PCX",
    "PPM",
    "SGI",
    "DDS",
    "ICO",
    "ICNS",
    "QOI",
    "XBM",
    "XPM",
}


def format_support() -> dict:
    return dict(
        accept=",".join(extension for _, extensions in FORMAT_GROUPS for extension in extensions),
        groups=[dict(name=name, extensions=list(extensions)) for name, extensions in FORMAT_GROUPS],
        ecw_available=ecw_reader.available(),
        multiple_frames="first",
        working_color="RGB8",
    )


def read_picture(data: bytes, *, scene: bool = False) -> Image.Image:
    if not data or len(data) > (MAX_DECODE_BYTES if scene else 15 * 1024 * 1024):
        raise UserError("حجم تصویر بیش از حد مجاز است.")
    maximum_pixels = MAX_PIXELS if scene else 16_000_000
    if ecw_reader.is_ecw(data):
        return ecw_reader.read_ecw(data, maximum_pixels)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in PIL_FORMATS:
                    raise UserError(
                        "این قالب تصویری پشتیبانی نمی‌شود. فهرست فرمت‌ها را در راهنمای برنامه ببینید."
                    )
                if source.width * source.height > maximum_pixels or max(source.size) > 16000:
                    raise UserError(
                        "تصویر بیش از حد بزرگ است؛ تصویر اصلی حداکثر ۶۴ میلیون پیکسل و ضلع ۱۶۰۰۰ دارد."
                    )
                notes = []
                if source.format == "PSD":
                    notes.append("تصویر ادغام‌شدهٔ PSD وارد شد؛ لایه‌ها جداگانه وارد نمی‌شوند.")
                elif getattr(source, "n_frames", 1) > 1:
                    notes.append("فایل چندصفحه‌ای یا متحرک بود؛ فقط صفحه یا فریم اول وارد شد.")
                    source.seek(0)
                if (
                    source.format == "TIFF"
                    and raster_reader.available()
                    and (max(source.tag_v2.get(258, (8,))) > 8 or source.tag_v2.get(277, 1) > 4)
                ):
                    result = raster_reader.read_raster(data, maximum_pixels)
                    result.info.setdefault("atlas_import_notes", []).extend(notes)
                    return result
                return normalize_open_picture(source)
    except UnidentifiedImageError as exc:
        if raster_reader.available():
            return raster_reader.read_raster(data, maximum_pixels)
        raise UserError(
            "فرمت یا کدک تصویر قابل خواندن نیست. از فهرست فرمت‌های راهنمای برنامه استفاده کنید؛ برای ECW بستهٔ کامل ویندوز لازم است."
        ) from exc
    except (
        OSError,
        ValueError,
        EOFError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise UserError("فایل تصویر ناقص، خراب یا دارای نوع فشرده‌سازی پشتیبانی‌نشده است.") from exc


def normalize_open_picture(source: Image.Image) -> Image.Image:
    notes = []
    if source.format == "PSD":
        notes.append("تصویر ادغام‌شدهٔ PSD وارد شد؛ لایه‌ها جداگانه وارد نمی‌شوند.")
    elif getattr(source, "n_frames", 1) > 1:
        notes.append("فایل چندصفحه‌ای یا متحرک بود؛ فقط صفحه یا فریم اول وارد شد.")
        source.seek(0)
    source.load()
    picture = ImageOps.exif_transpose(source)
    if picture.mode in ("I", "F") or picture.mode.startswith("I;16"):
        array = np.asarray(picture, dtype=np.float32)
        stride = max(1, int(np.ceil(np.sqrt(array.size / 1_000_000))))
        sample = array[::stride, ::stride]
        finite = sample[np.isfinite(sample)]
        low, high = np.percentile(finite, (0.5, 99.5)) if finite.size else (0, 0)
        if high > low:
            array = np.nan_to_num((array - low) * (255 / float(high - low)), nan=0, posinf=255, neginf=0)
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = np.zeros(array.shape, dtype=np.uint8)
        picture = Image.fromarray(array)
        notes.append("تصویر خاکستری با تنظیم خودکار دامنهٔ روشنایی به ۸ بیت تبدیل شد.")
    if picture.mode in ("RGBA", "LA", "PA") or (picture.mode == "P" and "transparency" in picture.info):
        rgba = picture.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        result = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        result = picture.convert("RGB")
    result.info["atlas_import_notes"] = notes
    return result
