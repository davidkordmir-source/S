from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import IMAGE_MODEL, IMAGE_REVISION, MODEL_CACHE, TEXT_MODEL, TEXT_REVISION
from .errors import UserError


class Engine:
    """Lazy CPU inference. No secret/API key or remote inference is required."""

    def __init__(self, root: Path):
        self.cache = root / MODEL_CACHE
        os.environ.setdefault("HF_HOME", str(self.cache))
        os.environ.setdefault("HF_XET_CACHE", str(self.cache / "xet"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        bundle = (
            Path(os.environ.get("ATLAS_MODEL_DIR", ""))
            if os.environ.get("ATLAS_MODEL_DIR")
            else (
                Path(sys._MEIPASS) / "models"
                if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[2] / ".model-bundle"
            )
        )
        self.paths = {}
        for kind, model, revision in (
            ("image", IMAGE_MODEL, IMAGE_REVISION),
            ("text", TEXT_MODEL, TEXT_REVISION),
        ):
            local = bundle / kind
            cached = self.cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
            self.paths[kind] = local if (local / "modules.json").is_file() else cached
        self._image: Any = None
        self._text: Any = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "image_loaded": self._image is not None,
            "text_loaded": self._text is not None,
            "image_model": IMAGE_MODEL,
            "text_model": TEXT_MODEL,
            "device": "CPU",
            "offline": True,
            "image_available": (self.paths["image"] / "modules.json").is_file(),
            "text_available": (self.paths["text"] / "modules.json").is_file(),
        }

    def _load(self, kind: str) -> Any:
        model = self._image if kind == "image" else self._text
        if model is not None:
            return model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except ImportError as exc:
            raise UserError("بسته‌های هوش مصنوعی نصب نیستند. راهنمای نصب ویندوز را اجرا کنید.") from exc
        directory = self.paths[kind]
        if not (directory / "modules.json").is_file():
            raise UserError("فایل‌های مدل آفلاین در بسته موجود نیستند. پوشهٔ کامل نسخهٔ آفلاین را استخراج کنید.")
        try:
            model = SentenceTransformer(
                str(directory),
                device="cpu",
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception as exc:
            raise UserError(
                "مدل آفلاین خوانده نشد. کامل‌بودن پوشه‌های برنامه و گزارش خطا را بررسی کنید."
            ) from exc
        if kind == "image":
            self._image = model
        else:
            self._text = model
        return model

    def prepare(self) -> None:
        with self._lock:
            self._load("image")
            self._load("text")

    def image(self, path: Path) -> np.ndarray:
        with Image.open(path) as source:
            return self.images([source.convert("RGB")])[0]

    def images(self, pictures: list[Image.Image]) -> np.ndarray:
        with self._lock:
            model = self._load("image")
            return np.asarray(
                model.encode(pictures, batch_size=8, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )

    def text(self, query: str) -> np.ndarray:
        with self._lock:
            model = self._load("text")
            # This multilingual model was distilled into CLIP's image space;
            # Persian is supported without translating through an API.
            return np.asarray(
                model.encode(query, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
            )
