"""Shape-only, multi-view exemplar matching with bounded reference image memory."""

from __future__ import annotations

import functools
import hashlib
import math
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image

from .config import DETAIL_REVISION, INDEX_VERSION
from .errors import UserError
from .visual_features import (
    appearance_scores,
    compare,
    dense_proposals,
    descriptor,
    model_picture,
    peaks,
    shape_picture,
    structural,
)

ANGLES = tuple(range(0, 360, 45))


class ReferenceMatcher:
    def __init__(self, similarity, settings, job):
        self.similarity, self.settings, self.job = similarity, settings, job
        self.views = settings["references"]
        self.angles = ANGLES if settings["rotate"] else (0,)
        self.scales = (0.5, 0.65, 0.8, 1, 1.25, 1.6, 2) if settings["broad"] else (0.8, 1, 1.25)
        self.sample = functools.lru_cache(maxsize=4)(self._sample)
        self.bank = []
        if settings["intelligent"]:
            for index, view in enumerate(self.views):
                job.update(
                    1, f"آماده‌سازی نمونهٔ آموزشی {index + 1} از {len(self.views)}؛ مقایسهٔ شکل و بافت بدون رنگ"
                )
                job.check()
                self.bank.append(self.features(view["id"]))

    def _sample(self, id):
        with Image.open(self.similarity.folder / f"{id}.png") as source:
            if np.asarray(source.convert("L"), dtype=np.float32).std() < 3:
                raise UserError(
                    "نمونه تقریباً فقط رنگ یکنواخت دارد؛ عکسی با لبه، شکل یا بافت مشخص بده یا این نمونه را غیرفعال کن."
                )
            return shape_picture(source)

    def features(self, id):
        folder = self.similarity.folder / "features"
        folder.mkdir(exist_ok=True)
        key = hashlib.sha256(
            f"{id}:{INDEX_VERSION}:{DETAIL_REVISION}:shape-v2:{self.angles}".encode()
        ).hexdigest()
        path = folder / f"{key}.npz"
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    values = {k: saved[k] for k in ("clip", "cls", "pooled", "spatial")}
                self.validate_features(values)
                return values
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, UserError):
                path.unlink(missing_ok=True)
        pictures = [
            self.sample(id).rotate(a, resample=Image.Resampling.BICUBIC, expand=True) for a in self.angles
        ]
        clip = np.asarray(self.similarity.workshop.engine.images(pictures), np.float32)
        self.job.check()
        details = [
            descriptor(f)
            for f in self.similarity.workshop.engine.details([model_picture(p) for p in pictures])
        ]
        values = dict(
            clip=clip,
            **{k: np.asarray([f[k] for f in details], np.float32) for k in ("cls", "pooled", "spatial")},
        )
        self.validate_features(values)
        temporary = folder / f"{key}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **values)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return values

    def validate_features(self, values):
        shapes = dict(
            clip=(len(self.angles), 512),
            cls=(len(self.angles), 384),
            pooled=(len(self.angles), 384),
            spatial=(len(self.angles), 9, 384),
        )
        if any(values[k].shape != s or not np.isfinite(values[k]).all() for k, s in shapes.items()):
            raise UserError("ویژگی‌های نمونهٔ آموزشی معتبر نیست؛ سلامت مدل را بررسی کن.")

    def templates(self, factor):
        for view_index, view in enumerate(self.views):
            sample = self.sample(view["id"])
            for scale in self.scales:
                size = (
                    max(8, round(view["width"] * scale / factor)),
                    max(8, round(view["height"] * scale / factor)),
                )
                resized = sample.resize(size, Image.Resampling.LANCZOS)
                for angle in self.angles:
                    rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                    valid = None
                    if angle % 90:
                        mask = Image.new("L", resized.size, 255).rotate(
                            angle, resample=Image.Resampling.BICUBIC, expand=True
                        )
                        valid = (np.asarray(mask) > 250).astype(np.uint8) * 255
                    yield view_index, np.asarray(rotated).copy(), angle, valid

    def proposals(self, picture, sx, sy, span, x0, y0, box, suppress):
        image = np.asarray(picture)
        edges = structural(image)
        candidates, by_view = [], {}
        for view, template, angle, valid in self.templates(self.settings["factor"]):
            self.job.check()
            th, tw = template.shape[:2]
            if th > image.shape[0] or tw > image.shape[1]:
                continue
            if template.astype(np.float32).std() < 3:
                continue
            gray = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=valid)
            template_edges = structural(template)
            if template_edges.std() < 1e-3:
                continue
            edge = cv2.matchTemplate(edges, template_edges, cv2.TM_CCOEFF_NORMED, mask=valid)
            gray, edge = (np.nan_to_num(v, nan=-1, posinf=-1, neginf=-1) for v in (gray, edge))
            values = np.maximum(0.75 * gray + 0.25 * edge, edge)
            lx = min(values.shape[1], max(0, math.ceil(min(span, box[2] - x0) / sx)))
            ly = min(values.shape[0], max(0, math.ceil(min(span, box[3] - y0) / sy)))
            for px, py, confidence in peaks(values[:ly, :lx], 0.35, min(tw, th) / 2):
                b = (
                    round(x0 + px * sx),
                    round(y0 + py * sy),
                    min(box[2], round(x0 + (px + tw) * sx)),
                    min(box[3], round(y0 + (py + th) * sy)),
                )
                by_view.setdefault(view, []).append(
                    (b, confidence, view, angle, float(gray[py, px]), float(edge[py, px]))
                )
        for proposals in by_view.values():
            candidates.extend(suppress(proposals))
        if self.bank:
            refs = [
                dict(cls=b["cls"][i], pooled=b["pooled"][i], spatial=b["spatial"][i])
                for b in self.bank
                for i in range(len(self.angles))
            ]
            features = []
            for bounds, confidence in dense_proposals(
                picture,
                self.similarity.workshop.engine,
                refs,
                self.settings["width"] / sx,
                self.settings["height"] / sy,
                self.scales,
                self.settings["rotate"],
                self.job.check,
                sizes=[(v["width"] / sx, v["height"] / sy) for v in self.views],
            ):
                px, py, px1, py1 = bounds
                if (
                    px < 0
                    or py < 0
                    or px1 > picture.width
                    or py1 > picture.height
                    or px * sx >= span
                    or py * sy >= span
                ):
                    continue
                b = (round(x0 + px * sx), round(y0 + py * sy), round(x0 + px1 * sx), round(y0 + py1 * sy))
                features.append((b, confidence, None, 0, None, None))
            candidates.extend(suppress(features, overlap=0.7))
        # Keep independent view proposals until common crop scoring; exact boxes
        # can be embedded once regardless of how many views proposed them.
        return list({item[0]: item for item in candidates}.values())

    def score(self, chunk, crops, scan, tile):
        engine = self.similarity.workshop.engine
        gray_crops = [shape_picture(crop) for crop in crops]
        clip = engine.images(gray_crops) if self.bank else None
        details = (
            [descriptor(f) for f in engine.details([model_picture(p) for p in gray_crops])]
            if self.bank
            else None
        )
        if clip is not None and (clip.shape != (len(chunk), 512) or not np.isfinite(clip).all()):
            raise UserError("مقایسهٔ هوشمند نمونه‌ها کامل نشد.")
        records = []
        for index, (item, crop) in enumerate(zip(chunk, gray_crops, strict=True)):
            bounds, _, proposed_view, angle, gray, edge = item
            scores = []
            for vi, view in enumerate(self.views):
                self.job.check()
                if vi == proposed_view:
                    appearance, structure, orientation = gray, edge, angle
                else:
                    appearance, structure, orientation = appearance_scores(
                        crop, self.sample(view["id"]), self.angles
                    )
                appearance, structure = float(np.clip(appearance, 0, 1)), float(np.clip(structure, 0, 1))
                visual = 0.4 * appearance + 0.6 * structure
                learned = semantic = layout = None
                score = visual if structure >= 0.25 else 0.0
                if self.bank:
                    bank = self.bank[vi]
                    refs = [
                        dict(cls=bank["cls"][r], pooled=bank["pooled"][r], spatial=bank["spatial"][r])
                        for r in range(len(self.angles))
                    ]
                    raw, best = compare(details[index], refs)
                    learned = max(0.0, 2 * (raw - 0.5))
                    spatial = float((details[index]["spatial"] * bank["spatial"][best]).sum(axis=-1).mean())
                    layout = float(np.clip((spatial - 0.4) / 0.6, 0, 1))
                    semantic = max(0.0, 2 * (float((bank["clip"] @ clip[index]).max()) - 0.5))
                    if max(structure, layout) >= 0.28:
                        score = max(
                            0.85 * visual + 0.15 * semantic, 0.8 * learned + 0.1 * layout + 0.1 * semantic
                        )
                scores.append((score, appearance, structure, learned, semantic, orientation, view["id"]))
            best = max(scores, key=lambda s: s[0])
            score, appearance, structure, learned, semantic, angle, reference = best
            support = sum(s[0] >= 0.55 for s in scores)
            b = bounds
            records.append(
                (
                    scan,
                    tile,
                    b[0],
                    b[1],
                    b[2] - b[0],
                    b[3] - b[1],
                    score,
                    appearance,
                    semantic,
                    angle,
                    learned,
                    structure,
                    reference,
                    support,
                )
            )
        return records
