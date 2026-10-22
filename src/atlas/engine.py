from __future__ import annotations

import os
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
        model_id = IMAGE_MODEL if kind == "image" else TEXT_MODEL
        revision = IMAGE_REVISION if kind == "image" else TEXT_REVISION
        try:
            # SentenceTransformers validates model downloads with the upstream
            # HTTPS/hub client. Never enable trust_remote_code or disable TLS.
            model = SentenceTransformer(
                model_id,
                device="cpu",
                cache_folder=str(self.cache),
                trust_remote_code=False,
                revision=revision,
            )
        except Exception as exc:
            raise UserError(
                "مدل هوش مصنوعی بارگیری نشد. اتصال به huggingface.co و فضای دیسک را بررسی کنید؛ سپس دوباره تلاش کنید."
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
        with self._lock:
            model = self._load("image")
            with Image.open(path) as source:
                picture = source.convert("RGB")
                return np.asarray(model.encode(picture, normalize_embeddings=True), dtype=np.float32)

    def text(self, query: str) -> np.ndarray:
        with self._lock:
            model = self._load("text")
            # This multilingual model was distilled into CLIP's image space;
            # Persian is supported without translating through an API.
            return np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)
