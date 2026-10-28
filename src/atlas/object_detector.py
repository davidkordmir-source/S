"""Native OWLv2 image-guided/text-guided detection with reusable scene embeddings."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
from scipy.special import expit

from .errors import UserError
from .visual_features import shape_picture, unit


class ObjectDetector:
    def __init__(self, engine, directory: Path):
        self.engine = engine
        self.directory = directory
        self.model = self.processor = None
        self.lock = threading.Lock()
        self.vision_calls = 0
        self.query_calls = 0

    def status(self):
        return dict(
            available=(self.directory / "model.safetensors").is_file(),
            loaded=self.model is not None,
            model="OWLv2 image/text-guided object detection",
            vision_calls=self.vision_calls,
            query_calls=self.query_calls,
        )

    def revision(self):
        marker = self.directory / "atlas-source.json"
        return hashlib.sha256(marker.read_bytes()).hexdigest() if marker.is_file() else "owlv2-local-v1"

    def _load(self):
        if self.model is not None:
            return self.model, self.processor
        if not self.status()["available"]:
            raise UserError("مدل تشخیص هم‌نوع‌ها در پوشهٔ برنامه نیست؛ بستهٔ کامل مدل‌های نسخهٔ جدید را نصب کن.")
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        torch.set_num_threads(self.engine.compute.cpu_threads())
        self.processor = Owlv2Processor.from_pretrained(
            str(self.directory), local_files_only=True, backend="pil"
        )
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(
                str(self.directory),
                local_files_only=True,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .eval()
            .to(self.engine.compute.torch_device())
        )
        return self.model, self.processor

    def _pixels(self, picture, resolution, colour):
        import torch

        _, processor = self._load()
        # This is the model's published aspect-preserving resize/square-pad path.
        image = picture.convert("RGB") if colour else shape_picture(picture)
        return processor.image_processor(
            images=image, size={"height": resolution, "width": resolution}, return_tensors="pt"
        )["pixel_values"].to(self.engine.compute.torch_device(), dtype=torch.float32)

    def scene(self, picture, resolution=768, colour=False):
        import torch

        with self.lock, torch.inference_mode():
            model, _ = self._load()
            feature_map, _ = model.image_embedder(
                self._pixels(picture, resolution, colour), interpolate_pos_encoding=True
            )
            features = feature_map.flatten(1, 2)
            _, classes = model.class_predictor(features)
            boxes = model.box_predictor(features, feature_map, interpolate_pos_encoding=True)
            shift = model.class_head.logit_shift(features)
            scale = model.class_head.elu(model.class_head.logit_scale(features)) + 1
            objectness = model.objectness_predictor(features)
            self.vision_calls += 1
            return dict(
                classes=unit(classes[0].float().cpu().numpy()),
                boxes=boxes[0].float().cpu().numpy(),
                shift=shift[0, :, 0].float().cpu().numpy(),
                scale=scale[0, :, 0].float().cpu().numpy(),
                objectness=objectness[0].float().cpu().numpy(),
            )

    def image_query(self, picture, resolution=768, colour=False):
        import torch

        with self.lock, torch.inference_mode():
            model, _ = self._load()
            feature_map, _ = model.image_embedder(
                self._pixels(picture, resolution, colour), interpolate_pos_encoding=True
            )
            features = feature_map.flatten(1, 2)
            embeddings, indexes, _ = model.embed_image_query(
                features, feature_map, interpolate_pos_encoding=True
            )
            self.query_calls += 1
            return unit(embeddings[0].float().cpu().numpy())

    def text_queries(self, phrases):
        import torch

        phrases = list(dict.fromkeys(p.strip() for p in phrases if p.strip()))
        if not phrases:
            return np.empty((0, 512), np.float32)
        with self.lock, torch.inference_mode():
            model, processor = self._load()
            maximum = model.config.text_config.max_position_embeddings
            # The VLM emits short free-form detector phrases; never a class allowlist.
            inputs = processor.tokenizer(
                phrases, padding="max_length", max_length=maximum, truncation=True, return_tensors="pt"
            )
            features = model.owlv2.get_text_features(
                **{k: v.to(self.engine.compute.torch_device()) for k, v in inputs.items()}
            )
            self.query_calls += 1
            return unit(features.pooler_output.float().cpu().numpy())

    @staticmethod
    def classify(scene, images=None, texts=None, negatives=None, text_required=False):
        """Exactly the published class head, evaluated on cached normalized embeddings.

        No template correlation, fixed spatial layout or exact-keypoint gate is used.
        Scores are model scores; no per-tile max-to-100% normalization is applied.
        """
        classes = scene["classes"]

        def scores(queries):
            if queries is None or not len(queries):
                return None
            cosine = np.einsum("nc,qc->nq", classes, unit(np.asarray(queries, np.float32)), optimize=False)
            return expit((cosine + scene["shift"][:, None]) * scene["scale"][:, None])

        image_matrix, text_matrix, negative_matrix = scores(images), scores(texts), scores(negatives)
        image_score = image_matrix.max(axis=1) if image_matrix is not None else None
        text_score = text_matrix.max(axis=1) if text_matrix is not None else None
        negative_score = negative_matrix.max(axis=1) if negative_matrix is not None else None
        if image_score is None and text_score is None:
            raise UserError("عکس نمونه یا توضیح قابل جست‌وجو لازم است.")
        score = (
            image_score
            if text_score is None
            else text_score
            if image_score is None
            else np.sqrt(image_score * text_score)
            if text_required
            else np.maximum(image_score, text_score)
        )
        if negative_score is not None:
            score = np.where(negative_score > score + 0.08, score * 0.15, score)
        return dict(
            scores=score,
            image_indexes=image_matrix.argmax(axis=1) if image_matrix is not None else None,
            support=(image_matrix >= np.maximum(0.01, image_score[:, None] * 0.8)).sum(axis=1)
            if image_matrix is not None
            else None,
            image_scores=image_score,
            text_scores=text_score,
            negative_scores=negative_score,
            objectness=expit(scene["objectness"]),
        )

    @staticmethod
    def boxes(scene, width, height):
        # OWLv2 pads on the bottom/right to a square before resizing.
        center = scene["boxes"][:, :2]
        sides = scene["boxes"][:, 2:]
        values = np.concatenate([center - sides / 2, center + sides / 2], axis=1) * max(width, height)
        values[:, [0, 2]] = np.clip(values[:, [0, 2]], 0, width)
        values[:, [1, 3]] = np.clip(values[:, [1, 3]], 0, height)
        return values

    def close(self):
        with self.lock:
            self.model = self.processor = None
