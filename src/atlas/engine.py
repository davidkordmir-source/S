from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .compute import Compute
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
    """Lazy local inference with hardware acceleration when supported."""

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
        self._aerial: Any = None
        self._segment: Any = None
        self._segment_embeddings = None
        self.paths["aerial"] = bundle / "aerial"
        self.paths["segment"] = bundle / "segment"
        self.compute = Compute(root, bundle / "detail-onnx/model.onnx")
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "image_loaded": self._image is not None,
            "text_loaded": self._text is not None,
            "image_model": IMAGE_MODEL,
            "text_model": TEXT_MODEL,
            "device": self.compute.effective,
            "compute": self.compute.status(),
            "offline": True,
            "image_available": (self.paths["image"] / "modules.json").is_file(),
            "text_available": (self.paths["text"] / "modules.json").is_file(),
            "detail_available": (self.paths["detail"] / "config.json").is_file(),
            "detail_loaded": self._detail is not None,
            "aerial_available": (self.paths["aerial"] / "model.safetensors").is_file(),
            "segmentation_available": (self.paths["segment"] / "model.safetensors").is_file(),
        }

    def _load(self, kind: str) -> Any:
        model = getattr(self, "_" + kind)
        if model is not None:
            return model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.compute.cpu_threads())
        except ImportError as exc:
            raise UserError("بسته‌های هوش مصنوعی نصب نیستند. راهنمای نصب ویندوز را اجرا کنید.") from exc
        directory = self.paths[kind]
        if not (
            directory / ("config.json" if kind in ("detail", "aerial", "segment") else "modules.json")
        ).is_file():
            raise UserError("فایل‌های مدل آفلاین در بسته موجود نیستند. پوشهٔ کامل نسخهٔ آفلاین را استخراج کنید.")
        try:
            if kind == "detail":
                from transformers import Dinov2Model

                model = (
                    Dinov2Model.from_pretrained(
                        str(directory),
                        local_files_only=True,
                        use_safetensors=True,
                        attn_implementation="sdpa",
                    )
                    .eval()
                    .to(self.compute.torch_device())
                )
            elif kind == "aerial":
                from transformers import CLIPVisionModelWithProjection

                model = (
                    CLIPVisionModelWithProjection.from_pretrained(
                        str(directory), local_files_only=True, use_safetensors=True, dtype=torch.float32
                    )
                    .eval()
                    .to(self.compute.torch_device())
                )
            elif kind == "segment":
                from transformers import Sam2Model

                model = (
                    Sam2Model.from_pretrained(
                        str(directory), local_files_only=True, use_safetensors=True, dtype=torch.float32
                    )
                    .eval()
                    .to(self.compute.torch_device())
                )
            else:
                model = SentenceTransformer(
                    str(directory),
                    device=self.compute.torch_device(),
                    trust_remote_code=False,
                    local_files_only=True,
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
            result = [None] * len(pictures)
            groups = {}
            for index, picture in enumerate(pictures):
                width, height = picture.size
                if max(width, height) > 448 or min(width, height) < 14 or width % 14 or height % 14:
                    raise UserError("اندازهٔ ورودی مقایسهٔ ویژگی‌ها معتبر نیست.")
                groups.setdefault((width, height), []).append((index, picture))
            for (width, height), items in groups.items():
                batch = 2 if max(width, height) > 224 else 8
                for start in range(0, len(items), batch):
                    chunk = items[start : start + batch]
                    pixels = np.stack(
                        [np.asarray(p.convert("RGB"), np.float32).transpose(2, 0, 1) / 255 for _, p in chunk]
                    )
                    pixels = (
                        pixels - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
                    ) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
                    values = self.compute.details(pixels)
                    if values is None:
                        model = self._load("detail")
                        values = model(
                            pixel_values=torch.from_numpy(pixels).to(self.compute.torch_device())
                        ).last_hidden_state
                        values = functional.normalize(values, dim=-1).cpu().numpy()
                    else:
                        values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
                    for (index, _), value in zip(chunk, values, strict=True):
                        result[index] = dict(
                            cls=value[0], grid=value[1:].reshape(height // 14, width // 14, -1)
                        )
            return result

    def aerial(self, pictures):
        import torch
        from PIL import ImageOps

        from .visual_features import shape_picture

        with self._lock, torch.inference_mode():
            model = self._load("aerial")
            result = []
            for start in range(0, len(pictures), 8):
                images = [
                    ImageOps.fit(shape_picture(p), (224, 224), method=Image.Resampling.BICUBIC)
                    for p in pictures[start : start + 8]
                ]
                pixels = np.stack([np.asarray(p, np.float32).transpose(2, 0, 1) / 255 for p in images])
                pixels = (
                    pixels - np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[None, :, None, None]
                ) / np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[None, :, None, None]
                vectors = model(
                    pixel_values=torch.from_numpy(pixels).to(self.compute.torch_device())
                ).image_embeds
                result.append(torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
            return np.concatenate(result) if result else np.empty((0, 512), np.float32)

    def segment(self, picture, points=None, box=None):
        import torch

        with self._lock, torch.inference_mode():
            model = self._load("segment")
            device = self.compute.torch_device()
            # SAM2 uses square resizing (as its published processor), rather than SAM1 padding.
            size = model.config.prompt_encoder_config.image_size
            rgb = picture.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            pixels = np.asarray(rgb, np.float32).transpose(2, 0, 1) / 255
            pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )[:, None, None]
            options = {}
            if points:
                options["input_points"] = torch.tensor(
                    [[[[p["x"] * size / picture.width, p["y"] * size / picture.height] for p in points]]],
                    device=device,
                    dtype=torch.float32,
                )
                options["input_labels"] = torch.tensor([[[int(p["label"]) for p in points]]], device=device)
            if box:
                options["input_boxes"] = torch.tensor(
                    [
                        [
                            [
                                box["x"] * size / picture.width,
                                box["y"] * size / picture.height,
                                (box["x"] + box["width"]) * size / picture.width,
                                (box["y"] + box["height"]) * size / picture.height,
                            ]
                        ]
                    ],
                    device=device,
                )
            import hashlib

            key = hashlib.sha256(str(picture.size).encode() + picture.tobytes()).hexdigest()
            if self._segment_embeddings is None or self._segment_embeddings[0] != key:
                embeddings = model.get_image_embeddings(torch.from_numpy(pixels)[None].to(device))
                self._segment_embeddings = (key, embeddings)
            output = model(image_embeddings=self._segment_embeddings[1], **options, multimask_output=True)
            best = int(output.iou_scores[0, 0].argmax())
            logits = output.pred_masks[0, 0, best][None, None]
            mask = (
                torch.nn.functional.interpolate(
                    logits, size=(picture.height, picture.width), mode="bilinear", align_corners=False
                )[0, 0]
                > 0
            )
            return Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255), float(
                output.iou_scores[0, 0, best]
            )

    def configure(self, device, performance=None):
        with self._lock:
            result = self.compute.configure(device, performance)
            import torch

            torch.set_num_threads(self.compute.cpu_threads())
            import cv2

            cv2.setNumThreads(min(8, self.compute.cpu_threads()))
            for name in ("image", "text", "detail", "aerial", "segment"):
                setattr(self, "_" + name, None)
            self._segment_embeddings = None
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
