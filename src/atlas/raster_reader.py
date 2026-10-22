"""Offline access to the bundled native GDAL/ECW decoder on Windows."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from .errors import UserError

DRIVERS = ("ECW", "GTiff", "HFA", "NITF", "JP2OpenJPEG", "JP2ECW", "PCIDSK")


def bundle_directory() -> Path:
    if os.environ.get("ATLAS_GDAL_DIR"):
        return Path(os.environ["ATLAS_GDAL_DIR"])
    return (
        Path(sys._MEIPASS) / "raster"
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2] / ".raster-bundle"
    )


def available() -> bool:
    return (
        os.name == "nt"
        and (bundle_directory() / "bin/gdalinfo.exe").is_file()
        and (bundle_directory() / "bin/gdal_translate.exe").is_file()
    )


def is_ecw(data: bytes) -> bool:
    return data[:3] == b"\x65\x02\x01" or data[4:8] == b"ecw "


def read_raster(data: bytes, maximum_pixels: int) -> Image.Image:
    if not available():
        raise UserError(
            "خواندن ECW و رسترهای تخصصی به بستهٔ کامل نسخهٔ جدید ویندوز نیاز دارد؛ پوشهٔ raster را همراه برنامه نگه دارید."
        )
    bundle = bundle_directory()
    environment = os.environ.copy()
    environment.update(
        PATH=str(bundle / "bin") + os.pathsep + environment.get("PATH", ""),
        GDAL_DRIVER_PATH=str(bundle / "apps/gdal/lib/gdalplugins"),
        GDAL_DATA=str(bundle / "apps/gdal/share/gdal"),
        PROJ_DATA=str(bundle / "share/proj"),
        PROJ_NETWORK="OFF",
        GDAL_PAM_ENABLED="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX="64",
        ECW_CACHE_MAXMEM="67108864",
        GDAL_NUM_THREADS="2",
        GDAL_VRT_ENABLE_RAWRASTERBAND="NO",
    )

    def execute(tool, arguments, timeout):
        result = subprocess.run(
            [str(bundle / "bin" / f"{tool}.exe"), *arguments],
            env=environment,
            capture_output=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            logging.error(
                "Native raster reader failed (%s): %s",
                result.returncode,
                result.stderr.decode("utf-8", errors="replace")[-3000:],
            )
            raise UserError(
                "فایل رستری خوانده نشد؛ ممکن است ناقص باشد یا به فایل‌های همراه نیاز داشته باشد. ECW، GeoTIFF، JP2، IMG، NITF و PIX تک‌فایلی پشتیبانی می‌شوند."
            )
        return result.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="atlas-raster-") as directory:
            source = Path(directory) / ("source.ecw" if is_ecw(data) else "source.dat")
            source.write_bytes(data)
            allowed = [value for driver in DRIVERS for value in ("-if", driver)]
            information = json.loads(
                execute("gdalinfo", ["-json", "-nomd", "-noct", "-norat", *allowed, str(source)], 45)
            )
            width, height = information["size"]
            if width * height > maximum_pixels or max(width, height) > 16000 or min(width, height) < 1:
                raise UserError(
                    "ابعاد تصویر بیش از حد مجاز است؛ تصویر اصلی حداکثر ۶۴ میلیون پیکسل و ضلع ۱۶۰۰۰ دارد. یک بخش کوچک‌تر از رستر را وارد کنید."
                )
            bands = information["bands"]
            if not bands:
                raise UserError("هیچ باند تصویری در فایل پیدا نشد.")
            notes = []
            colors = {band.get("colorInterpretation"): number + 1 for number, band in enumerate(bands)}
            selected = (
                [colors[color] for color in ("Red", "Green", "Blue")]
                if all(color in colors for color in ("Red", "Green", "Blue"))
                else ([1, 2, 3] if len(bands) >= 3 else [1])
            )
            if len(bands) > 4:
                notes.append("از رستر چندباندی، سه باند رنگی یا سه باند اول برای پردازش انتخاب شدند.")
            arguments = ["-of", "PNG", "-ot", "Byte", *allowed]
            for number in selected:
                arguments += ["-b", str(number)]
            if any(bands[number - 1]["type"] != "Byte" for number in selected):
                arguments += ["-scale", "-exponent", "1"]
                notes.append("دامنهٔ مقادیر رستر برای نمایش و پردازش به ۸ بیت تبدیل شد.")
            if len(selected) == 1 and bands[selected[0] - 1].get("colorInterpretation") == "Palette":
                arguments += ["-expand", "rgb"]
            output = Path(directory) / "decoded.png"
            execute("gdal_translate", [*arguments, str(source), str(output)], 180)
            if not output.is_file() or output.stat().st_size > maximum_pixels * 5 + 1024 * 1024:
                raise UserError("خروجی مبدل رستر معتبر نیست.")
            with Image.open(io.BytesIO(output.read_bytes())) as picture:
                if picture.size != (width, height):
                    raise UserError("ابعاد خروجی مبدل رستر معتبر نیست.")
                result = picture.convert("RGB")
            result.info["atlas_import_notes"] = notes
            return result
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError) as exc:
        raise UserError("خواندن رستر کامل نشد. فایل را بررسی کن یا بخش کوچک‌تری از آن را وارد کن.") from exc
