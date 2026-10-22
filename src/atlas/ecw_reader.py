"""Independent MIT-licensed ECW decoder; no proprietary SDK is used."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

from PIL import Image

from .errors import UserError


def available() -> bool:
    return importlib.util.find_spec("ecw2tiff") is not None


def is_ecw(data: bytes) -> bool:
    return data[:3] == b"\x65\x02\x01" or data[4:8] == b"ecw "


def read_ecw(data: bytes, maximum_pixels: int) -> Image.Image:
    os.environ.setdefault("RAYON_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
    try:
        from ecw2tiff import EcwReader

        with tempfile.TemporaryDirectory(prefix="atlas-ecw-") as directory:
            source = Path(directory) / "image.ecw"
            source.write_bytes(data)
            reader = None
            try:
                reader = EcwReader(str(source))
                information = reader.info
                width, height = information["width"], information["height"]
                if min(width, height) < 1 or max(width, height) > 16000 or width * height > maximum_pixels:
                    raise UserError(
                        "ابعاد تصویر بیش از حد مجاز است؛ تصویر اصلی حداکثر ۶۴ میلیون پیکسل و ضلع ۱۶۰۰۰ دارد. یک بخش کوچک‌تر از ECW را وارد کنید."
                    )
                raw, w, h, channels = reader.tile(0, 0, width, height, encode="raw")
                if (w, h) != (width, height) or channels not in (1, 3, 4) or len(raw) != w * h * channels:
                    raise UserError("خروجی خوانندهٔ ECW معتبر نیست.")
                image = Image.frombytes({1: "L", 3: "RGB", 4: "RGBA"}[channels], (w, h), raw).convert("RGB")
                image.info["atlas_import_notes"] = []
                return image
            finally:
                del reader
    except UserError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        # PyO3 exposes Rust panics as a BaseException. Keep a malformed input
        # from leaving the background job permanently in its running state.
        raise UserError(
            "فایل ECW خوانده نشد؛ فایل ممکن است ناقص باشد یا از گونهٔ کدگذاری پشتیبانی‌نشده استفاده کند. این نسخه روی ECW نسخهٔ ۲ و ۳ با تصویر RGB آزمایش شده است."
        ) from exc
