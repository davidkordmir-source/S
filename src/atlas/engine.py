from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    DETAIL_MODEL,
    DETAIL_REVISION,
    IMAGE_MODEL,
    IMAGE_REVISION,
    MODEL_CACHE,
    TEXT_MODEL,
    TEXT_REVISION,
)
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
            ("detail", DETAIL_MODEL, DETAIL_REVISION),
        ):
            local = bundle / kind
            cached = self.cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
            marker = "config.json" if kind == "detail" else "modules.json"
            self.paths[kind] = local if (local / marker).is_file() else cached
        self._image: Any = None
        self._text: Any = None
        self._detail: Any = None
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
            "detail_available": (self.paths["detail"] / "config.json").is_file(),
            "detail_loaded": self._detail is not None,
        }

    def _load(self, kind: str) -> Any:
        model = getattr(self, "_" + kind)
        if model is not None:
            return model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except ImportError as exc:
            raise UserError("بسته‌های هوش مصنوعی نصب نیستند. راهنمای نصب ویندوز را اجرا کنید.") from exc
        directory = self.paths[kind]
        if not (directory / ("config.json" if kind == "detail" else "modules.json")).is_file():
            raise UserError("فایل‌های مدل آفلاین در بسته موجود نیستند. پوشهٔ کامل نسخهٔ آفلاین را استخراج کنید.")
        try:
            if kind == "detail":
                from transformers import Dinov2Model

                model = Dinov2Model.from_pretrained(
                    str(directory), local_files_only=True, use_safetensors=True, attn_implementation="sdpa"
                ).eval()
            else:
                model = SentenceTransformer(
                    str(directory), device="cpu", trust_remote_code=False, local_files_only=True
                )
        except Exception as exc:
            raise UserError(
                "مدل آفلاین خوانده نشد. کامل‌بودن پوشه‌های برنامه و گزارش خطا را بررسی کنید."
            ) from exc
        setattr(self, "_" + kind, model)
        return model

    def details(self, pictures: list[Image.Image]) -> list[dict]:
        """Local DINOv2 patch descriptors. Inputs use multiples of 14, at most 448 px.

        Unlike global CLIP reranking these dense features propose locations even
        when their pixels do not correlate with the reference screenshot.
        """
        import torch
        import torch.nn.functional as functional

        with self._lock, torch.inference_mode():
            model = self._load("detail")
            result = []
            for picture in pictures:
                width, height = picture.size
                if max(width, height) > 448 or min(width, height) < 14 or width % 14 or height % 14:
                    raise UserError("اندازهٔ ورودی مقایسهٔ ویژگی‌ها معتبر نیست.")
                pixels = (
                    torch.from_numpy(np.asarray(picture.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
                )
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                values = model(pixel_values=((pixels - mean) / std)[None]).last_hidden_state[0]
                values = functional.normalize(values, dim=-1).cpu().numpy()
                result.append(dict(cls=values[0], grid=values[1:].reshape(height // 14, width // 14, -1)))
            return result

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
