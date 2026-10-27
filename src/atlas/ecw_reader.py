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
    return data[:3] in (b"\x65\x01\x01", b"\x65\x02\x01") or data[4:8] == b"ecw "


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


def inspect_ecw_layout(path: Path):
    """Validate small metadata records; directory size no longer limits scenes."""
    import struct

    with path.open("rb") as file:
        size = path.stat().st_size

        def at(position, length):
            if position < 0 or length > 1024 * 1024 or position + length > size:
                raise UserError("ساختار ECW معتبر نیست یا فرادادهٔ آن بیش از بودجهٔ خواننده است.")
            file.seek(position)
            result = file.read(length)
            if len(result) != length:
                raise UserError("فرادادهٔ ECW ناقص است.")
            return result

        header = at(0, 22)
        if header[0] == ord("e"):
            levels, bands = header[4], int.from_bytes(header[14:16], "big")
            bx, by = struct.unpack(">HH", header[18:22])
            position = 89 if header[1] > 1 else 22
        else:
            image_header, codestream = None, None

            def boxes(start, end):
                count = 0
                while start + 8 <= end:
                    count += 1
                    if count > 4096:
                        raise UserError("تعداد رکوردهای ECW بیش از بودجهٔ خواننده است.")
                    raw = at(start, 8)
                    length, kind = struct.unpack(">I4s", raw)
                    prefix = 8
                    if length == 1:
                        length = int.from_bytes(at(start + 8, 8), "big")
                        prefix = 16
                    elif length == 0:
                        length = end - start
                    if length < prefix or start + length > end:
                        raise UserError("ساختار ECW معتبر نیست.")
                    yield kind, start + prefix, start + length
                    start += length

            for kind, start, end in boxes(0, size):
                if kind == b"ecwh":
                    for child, first, last in boxes(start, end):
                        if child == b"ihdr":
                            image_header = at(first, last - first)
                elif kind == b"ecwc":
                    codestream = start
                elif kind == b"gtif":
                    at(start, end - start)  # native reader loads this metadata
            if image_header is None or len(image_header) < 22 or codestream is None:
                raise UserError("فرادادهٔ ECW کامل نیست.")
            levels = image_header[13]
            bands = int.from_bytes(image_header[11:13], "big")
            bx, by = struct.unpack(">HH", image_header[15:19])
            position = codestream
        if not (1 <= levels <= 32 and 1 <= bands <= 16 and bx and by):
            raise UserError("این ساختار ECW در خوانندهٔ فعلی پشتیبانی نمی‌شود.")
        total = 0
        for _ in range(levels):
            record = at(position, 9 + 4 * bands)
            width, height = struct.unpack(">II", record[1:9])
            total += ((width + bx - 1) // bx) * ((height + by - 1) // by)
            position += len(record)
        if header[0] == ord("e"):
            packed = int.from_bytes(at(position, 4), "big")
            if packed < 1 or position + 4 + packed > size:
                raise UserError("جدول بلوک ECW ناقص است.")
        return {"blocks": total, "directory_bytes": (total + 1) * 8 if header[0] == ord("e") else total * 12}
