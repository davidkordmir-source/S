"""Shape-only, multi-view exemplar matching with bounded reference image memory."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image

from . import geometry
from .config import AERIAL_REVISION, DETAIL_REVISION, INDEX_VERSION
from .errors import UserError
from .target_learning import TargetLearning
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
from .workshop import intersection_over_union

ANGLES = tuple(range(0, 360, 45))


class ReferenceMatcher:
    def __init__(self, similarity, settings, job):
        self.similarity, self.settings, self.job = similarity, settings, job
        self.views = settings["references"]
        self.angles = ANGLES if settings["rotate"] else (0,)
        self.scales = (0.5, 0.65, 0.8, 1, 1.25, 1.6, 2) if settings["broad"] else (0.8, 1, 1.25)
        self.sample = functools.lru_cache(maxsize=4)(self._sample)
        self.bank = []
        self.negative_bank = []
        self.negative_ids = {r["id"] for r in settings.get("negative_references", [])}
        self.exclusions = (
            similarity.learning.exclusions(settings["group"], settings["scene"])
            if settings.get("group") and settings.get("scene")
            else []
        )
        if settings["intelligent"]:
            for index, view in enumerate(self.views):
                job.update(
                    1, f"آماده‌سازی نمونهٔ آموزشی {index + 1} از {len(self.views)}؛ مقایسهٔ شکل و بافت بدون رنگ"
                )
                job.check()
                self.bank.append(self.features(view["id"]))
            for view in settings.get("negative_references", []):
                job.check()
                self.negative_bank.append(self.features(view["id"]))

    def _sample(self, id):
        with Image.open(self.similarity.folder / f"{id}.png") as source:
            if id not in self.negative_ids and np.asarray(source.convert("L"), dtype=np.float32).std() < 3:
                raise UserError(
                    "نمونه تقریباً فقط رنگ یکنواخت دارد؛ عکسی با لبه، شکل یا بافت مشخص بده یا این نمونه را غیرفعال کن."
                )
            return shape_picture(source)

    def features(self, id):
        folder = self.similarity.folder / "features"
        folder.mkdir(exist_ok=True)
        key = hashlib.sha256(
            f"{id}:{INDEX_VERSION}:{DETAIL_REVISION}:{AERIAL_REVISION}:precision-v1:{self.angles}".encode()
        ).hexdigest()
        path = folder / f"{key}.npz"
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    values = {
                        k: saved[k]
                        for k in (
                            "clip",
                            "cls",
                            "pooled",
                            "spatial",
                            "aerial",
                            "geo_points",
                            "geo_desc",
                            "geo_size",
                        )
                    }
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
        values["aerial"] = np.asarray(self.similarity.workshop.engine.aerial(pictures), np.float32)
        geo = self.sample(id).copy()
        geo.thumbnail((384, 384), Image.Resampling.LANCZOS)
        mask = self.similarity.segmentation.mask(id, geo.size)
        points, descriptors = geometry.local_features(geo, mask)
        values.update(
            geo_points=points,
            geo_desc=descriptors if descriptors is not None else np.empty((0, 128), np.float32),
            geo_size=np.asarray(geo.size, np.int32),
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
            aerial=(len(self.angles), 512),
        )
        if any(values[k].shape != s or not np.isfinite(values[k]).all() for k, s in shapes.items()):
            raise UserError("ویژگی‌های نمونهٔ آموزشی معتبر نیست؛ سلامت مدل را بررسی کن.")
        if (
            values["geo_points"].shape != (len(values["geo_desc"]), 2)
            or values["geo_desc"].shape[1:] != (128,)
            or values["geo_size"].shape != (2,)
        ):
            raise UserError("کش هندسی نمونه معتبر نیست.")

    def templates(self, factor):
        for view_index, view in enumerate(self.views):
            sample = self.sample(view["id"])
            for scale in self.scales:
                size = (
                    max(8, round(view["width"] * scale / factor)),
                    max(8, round(view["height"] * scale / factor)),
                )
                resized = sample.resize(size, Image.Resampling.LANCZOS)
                foreground = self.similarity.segmentation.mask(view["id"], size)
                for angle in self.angles:
                    rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                    valid = None
                    if foreground is not None or angle % 90:
                        mask = (
                            Image.fromarray(foreground)
                            if foreground is not None
                            else Image.new("L", resized.size, 255)
                        ).rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
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
            context = f"{self.settings.get('scene_fingerprint')}:{x0}:{y0}:{sx}:{sy}:{picture.size}:gray-v1"

            def provider(prepared, extent):
                return self.similarity.feature_store.cached(
                    context + ":" + str(extent) + ":" + str(prepared.size), prepared
                )

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
                feature_provider=provider if self.settings.get("scene_fingerprint") else None,
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
        if self.settings.get("fine_pose"):
            for view in self.views:
                size = (
                    max(8, round(view["width"] / self.settings["factor"])),
                    max(8, round(view["height"] / self.settings["factor"])),
                )
                reference = self.sample(view["id"]).resize(size, Image.Resampling.LANCZOS)
                mask = self.similarity.segmentation.mask(view["id"], size)
                poses = []
                for pose in geometry.pose_proposals(
                    reference, mask, picture, self.job.check, 3 if self.settings["broad"] else 1.6
                ):
                    q = pose["quad"]
                    px, py = q.min(axis=0)
                    px1, py1 = q.max(axis=0)
                    if (
                        min(px, py) < 0
                        or px1 > picture.width
                        or py1 > picture.height
                        or px * sx >= span
                        or py * sy >= span
                    ):
                        continue
                    b = (round(x0 + px * sx), round(y0 + py * sy), round(x0 + px1 * sx), round(y0 + py1 * sy))
                    poses.append((b, pose["score"], None, 0, None, None))
                candidates.extend(suppress(poses, overlap=0.7))
        # Keep independent view proposals until common crop scoring; exact boxes
        # can be embedded once regardless of how many views proposed them.
        return list({item[0]: item for item in candidates}.values())

    def score(self, chunk, crops, scan, tile):
        engine = self.similarity.workshop.engine
        valid = []
        for item, picture in zip(chunk, crops, strict=True):
            crop = shape_picture(picture)
            low, high = crop.getextrema()[0]
            # Normalized template correlation is undefined for a constant
            # candidate. OpenCV can report a spurious high edge score on some
            # CPU paths. Such a crop contains no shape/texture evidence at all.
            if low != high:
                valid.append((item, crop))
        if not valid:
            return []
        chunk, gray_crops = map(list, zip(*valid, strict=True))
        clip = engine.images(gray_crops) if self.bank else None
        details = (
            [descriptor(f) for f in engine.details([model_picture(p) for p in gray_crops])]
            if self.bank
            else None
        )
        aerial = (
            engine.aerial(gray_crops)
            if self.bank and (self.settings.get("aerial", True) or self.settings.get("adapter"))
            else None
        )
        if clip is not None and (clip.shape != (len(chunk), 512) or not np.isfinite(clip).all()):
            raise UserError("مقایسهٔ هوشمند نمونه‌ها کامل نشد.")
        records = []
        for index, (item, crop) in enumerate(zip(chunk, gray_crops, strict=True)):
            bounds, _, proposed_view, angle, gray, edge = item
            candidate_geo = geometry.local_features(crop) if self.settings.get("geometry") else None
            scores = []
            for vi, view in enumerate(self.views):
                self.job.check()
                if vi == proposed_view:
                    appearance, structure, orientation = gray, edge, angle
                else:
                    appearance, structure, orientation = appearance_scores(
                        crop,
                        self.sample(view["id"]),
                        self.angles,
                        self.similarity.segmentation.mask(view["id"], self.sample(view["id"]).size),
                    )
                appearance, structure = float(np.clip(appearance, 0, 1)), float(np.clip(structure, 0, 1))
                visual = 0.4 * appearance + 0.6 * structure
                learned = semantic = layout = None
                satellite = None
                proof = dict(status="disabled", score=None, inliers=0, rotation=None, scale=None, quad=None)
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
                    if aerial is not None and self.settings.get("aerial", True):
                        satellite = float(
                            np.clip(2 * (float((bank["aerial"] @ aerial[index]).max()) - 0.5), 0, 1)
                        )
                        if max(structure, layout) >= 0.28:
                            score = max(score, 0.75 * learned + 0.1 * layout + 0.15 * satellite)
                if candidate_geo is not None:
                    if self.bank:
                        proof = geometry.verify_features(
                            bank["geo_points"],
                            bank["geo_desc"],
                            tuple(bank["geo_size"]),
                            *candidate_geo,
                            crop.size,
                        )
                    else:
                        reference = self.sample(view["id"]).copy()
                        reference.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        proof = geometry.verify(
                            reference, crop, self.similarity.segmentation.mask(view["id"], reference.size)
                        )
                    if proof["status"] == "verified":
                        reference_image = self.sample(view["id"]).copy()
                        reference_image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        aligned = geometry.rectify(reference_image, crop, proof["quad"])
                        gray_aligned, edges_aligned, _ = appearance_scores(
                            aligned,
                            reference_image,
                            (0,),
                            self.similarity.segmentation.mask(view["id"], reference_image.size),
                        )
                        aligned_score = 0.4 * max(0, gray_aligned) + 0.6 * max(0, edges_aligned)
                        score = max(score, 0.8 * max(score, aligned_score) + 0.2 * proof["score"])
                    elif self.settings.get("strict_geometry"):
                        score = 0.0
                scores.append(
                    (
                        score,
                        appearance,
                        structure,
                        learned,
                        semantic,
                        orientation,
                        view["id"],
                        satellite,
                        proof,
                    )
                )
            best = max(scores, key=lambda s: s[0])
            score, appearance, structure, learned, semantic, angle, reference, satellite, proof = best
            adapted = None
            if self.bank and self.settings.get("adapter") and aerial is not None:
                adapted = TargetLearning.predict(
                    self.settings["adapter"], details[index]["pooled"], aerial[index]
                )
                score = 0.7 * score + 0.3 * adapted
                if adapted < 0.3:
                    score = min(score, 0.35)
            if self.negative_bank and details is not None:
                negative = max(
                    compare(
                        details[index],
                        [
                            dict(cls=b["cls"][r], pooled=b["pooled"][r], spatial=b["spatial"][r])
                            for r in range(len(self.angles))
                        ],
                    )[0]
                    for b in self.negative_bank
                )
                positive = max(
                    compare(
                        details[index],
                        [
                            dict(cls=b["cls"][r], pooled=b["pooled"][r], spatial=b["spatial"][r])
                            for r in range(len(self.angles))
                        ],
                    )[0]
                    for b in self.bank
                )
                if negative >= 0.8 and negative > positive + 0.04:
                    score = min(score, 0.35)
            for excluded in self.exclusions:
                e = (
                    excluded["x"],
                    excluded["y"],
                    excluded["x"] + excluded["width"],
                    excluded["y"] + excluded["height"],
                )
                if intersection_over_union(bounds, e) > 0.5:
                    score = 0.0
                    break
            support = sum(s[0] >= 0.55 for s in scores)
            b = bounds
            relative_scale = None
            if proof["scale"] is not None:
                matched_view = next(v for v in self.views if v["id"] == reference)
                reference_size = self.sample(reference).copy()
                reference_size.thumbnail((384, 384), Image.Resampling.LANCZOS)
                relative_scale = (
                    proof["scale"]
                    * (
                        ((b[2] - b[0]) * (b[3] - b[1]) / (crop.width * crop.height))
                        * (
                            reference_size.width
                            * reference_size.height
                            / (matched_view["width"] * matched_view["height"])
                        )
                    )
                    ** 0.5
                )
            quad = None
            if proof["quad"] is not None:
                quad = json.dumps(
                    [
                        [
                            b[0] + float(x) / crop.width * (b[2] - b[0]),
                            b[1] + float(y) / crop.height * (b[3] - b[1]),
                        ]
                        for x, y in proof["quad"]
                    ]
                )
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
                    satellite,
                    proof["score"],
                    proof["status"],
                    proof["inliers"],
                    proof["rotation"],
                    relative_scale,
                    quad,
                    adapted,
                )
            )
        return records
