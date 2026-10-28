"""Open-vocabulary and image-guided instance search, independent of pixel templates."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid

import numpy as np
from PIL import Image

from .errors import UserError
from .workshop import box_tuple, intersection_over_union, rect_dict


class SemanticSearch:
    def __init__(self, similarity):
        self.similar = similarity
        self.workshop = similarity.workshop
        self.engine = self.workshop.engine

    def plan(
        self,
        scene,
        reference=None,
        group=None,
        instruction=None,
        excluded_description=None,
        prepared=None,
        quality="balanced",
        scope=None,
        mode="complete",
        **ignored,
    ):
        info = self.workshop.scene(scene)
        self.workshop.scene_path(scene)
        profile = self.similar.library.group(group) if group else None
        if profile and profile["archived"]:
            raise UserError("این هدف حذف شده است؛ هدف فعال را انتخاب کن.")
        references = (
            [
                r
                for r in profile["references"]
                if r["enabled"] and r["label"] == 1 and r["partition"] == "train"
            ]
            if profile
            else [self.similar.reference(reference)]
            if reference
            else []
        )
        negative = (
            [
                r
                for r in profile["references"]
                if r["enabled"] and r["label"] == 0 and r["partition"] == "train"
            ]
            if profile
            else []
        )
        instruction = (
            instruction if instruction is not None else profile["description"] if profile else ""
        ).strip()
        if not references and not instruction:
            raise UserError("یک عکس نمونه بده یا خواسته‌ات را برای هوش مصنوعی بنویس.")
        if quality not in ("fast", "balanced", "detailed"):
            raise UserError("کیفیت پردازش معتبر نیست.")
        if mode != "complete":
            raise UserError(
                "جست‌وجوی هوشمند هم‌نوع‌ها از کش اختصاصی خودش استفاده می‌کند؛ حالت بررسی کامل را انتخاب کن."
            )
        box = (
            box_tuple(scope, info["width"], info["height"])
            if scope
            else (0, 0, info["width"], info["height"])
        )
        levels = [1] if quality == "fast" else [1, 2] if quality == "balanced" else [1, 2, 4]
        grids = []
        for factor in levels:
            span = 640 * factor
            columns = math.ceil((box[2] - box[0]) / span)
            rows = math.ceil((box[3] - box[1]) / span)
            grids.append(dict(factor=factor, span=span, columns=columns, rows=rows, count=columns * rows))
        settings = dict(
            method="similar",
            match_mode="instances",
            scene=scene,
            group=group,
            reference=references[0]["id"] if references else None,
            references=[
                dict(id=r["id"], name=r["name"], width=r["width"], height=r["height"]) for r in references
            ],
            negative_references=[dict(id=r["id"]) for r in negative],
            instruction=instruction,
            excluded_description=(
                excluded_description
                if excluded_description is not None
                else profile["excluded_description"]
                if profile
                else ""
            ),
            prepared=prepared,
            quality=quality,
            scope=rect_dict(box),
            mode="complete",
            grids=grids,
            windows=sum(g["count"] for g in grids),
            reference_count=len(references),
            width=references[0]["width"] if references else 128,
            height=references[0]["height"] if references else 128,
            scene_fingerprint=self.similar.feature_store.fingerprint(scene),
            detector_revision=self.engine.objects.revision(),
            assistant_revision=self.engine.assistant.revision(),
            algorithm="native-owl-instance-v1",
            colour_evidence=False,
        )
        scan = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with self.workshop.connect() as db:
            saved = db.execute("SELECT * FROM similar_scans WHERE id=?", (scan,)).fetchone()
        if saved:
            settings = json.loads(saved["settings"])
        return dict(
            allowed=True,
            scan=scan,
            settings=settings,
            windows=settings["windows"],
            completed=saved["next_tile"] if saved else 0,
            scope=settings["scope"],
        )

    def _image(self, reference, region=None):
        with Image.open(self.similar.folder / f"{reference}.png") as source:
            picture = source.convert("RGB")
        if region:
            bounds = tuple(
                round(v * (picture.width if i % 2 == 0 else picture.height)) for i, v in enumerate(region)
            )
            if bounds[2] - bounds[0] >= 8 and bounds[3] - bounds[1] >= 8:
                picture = picture.crop(bounds)
        return picture

    def _query(self, picture, colour, resolution):
        data = np.asarray(picture.convert("RGB"))
        key = hashlib.sha256(
            str((picture.size, colour, resolution)).encode()
            + data.tobytes()
            + self.engine.objects.revision().encode()
        ).hexdigest()
        folder = self.similar.folder / "object-queries"
        folder.mkdir(exist_ok=True)
        path = folder / f"{key}.npy"
        if path.is_file():
            try:
                vectors = np.load(path, allow_pickle=False)
                if vectors.shape != (1, 512) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid query cache")
                return vectors
            except (OSError, ValueError, EOFError):
                path.unlink(missing_ok=True)
        vectors = self.engine.objects.image_query(picture, resolution, colour)
        temporary = path.with_name(path.name + "-" + uuid.uuid4().hex + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        temporary.replace(path)
        return vectors

    def search(self, job, threshold=0.12, **options):
        if not math.isfinite(threshold) or not 0.01 <= threshold <= 0.99:
            raise UserError("حداقل امتیاز مدل باید بین یک و ۹۹ درصد باشد.")
        plan = self.plan(**options)
        settings, scan = plan["settings"], plan["scan"]
        if plan["completed"] == plan["windows"]:
            settings["reused_scan"] = True
            return self.similar.finish(job, settings, scan, threshold)
        references = settings["references"]
        if "assistant" not in settings:
            pictures = [self._image(r["id"]) for r in references[:4]]
            if not pictures:
                # Bind text-only/chat plans to the same actual scene overview.
                # The original large image is still read in bounded detector tiles.
                with Image.open(self.workshop.images / f"{settings['scene']}-preview.jpg") as preview:
                    pictures.append(preview.convert("RGB"))
            message = settings["instruction"]
            if settings["excluded_description"]:
                message += "\nموارد نامطلوب: " + settings["excluded_description"]
            understood = self.engine.assistant.understand(
                job, message, pictures, search=True, prepared=settings.get("prepared")
            )
            if understood["intent"] == "clarify" or not references and not understood["detector_phrases"]:
                raise UserError(understood["answer_fa"])
            settings["assistant"] = understood
            settings["colour_evidence"] = understood["use_colour"]
            # The language model is not run on every scene tile. Free its RAM
            # before the detector starts; a repeated instruction reuses its plan.
        self.engine.assistant.close()
        understood = settings["assistant"]
        colour = understood["use_colour"]
        resolution = {"fast": 512, "balanced": 768, "detailed": 960}[settings["quality"]]
        images, image_ids = [], []
        regions = understood.get("reference_regions", [])
        for i, reference in enumerate(references):
            job.update(2, f"آماده‌سازی نمونهٔ تصویری {i + 1} از {len(references)}")
            job.check()
            images.extend(
                self._query(
                    self._image(reference["id"], regions[i] if i < len(regions) else None), colour, resolution
                )
            )
            image_ids.append(reference["id"])
        image_vectors = np.array(images, np.float32) if images else None
        text_vectors = self.engine.objects.text_queries(understood["detector_phrases"])
        negative_vectors = list(self.engine.objects.text_queries(understood["excluded_phrases"]))
        for negative in settings["negative_references"]:
            picture = self._image(negative["id"])
            if np.asarray(picture.convert("L"), np.float32).std() >= 3:
                negative_vectors.extend(self._query(picture, colour, resolution))
        negative_vectors = np.asarray(negative_vectors, np.float32) if negative_vectors else None
        exclusions = (
            self.similar.learning.exclusions(settings["group"], settings["scene"])
            if settings["group"]
            else []
        )
        with self.workshop.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO similar_scans VALUES(?,?,?,?,?,?,?)",
                (
                    scan,
                    settings["scene"],
                    settings["reference"],
                    json.dumps(settings),
                    0,
                    settings["windows"],
                    time.time(),
                ),
            )
        store = self.similar.feature_store
        start_hits, start_misses = store.object_hits, store.object_misses
        from .similarity import local_suppress

        with self.workshop.reader(settings["scene"]) as reader:
            offset = 0
            for grid in settings["grids"]:
                for position in range(grid["count"]):
                    tile = offset + position
                    if tile < plan["completed"]:
                        continue
                    job.update(
                        3 + tile / settings["windows"] * 86,
                        f"تشخیص هوشمند هم‌نوع‌ها: بخش {tile + 1:,} از {settings['windows']:,}",
                    )
                    x0 = settings["scope"]["x"] + position % grid["columns"] * grid["span"]
                    y0 = settings["scope"]["y"] + position // grid["columns"] * grid["span"]
                    right = settings["scope"]["x"] + settings["scope"]["width"]
                    bottom = settings["scope"]["y"] + settings["scope"]["height"]
                    margin = 160 * grid["factor"]
                    outer = (
                        max(settings["scope"]["x"], x0 - margin),
                        max(settings["scope"]["y"], y0 - margin),
                        min(right, x0 + grid["span"] + margin),
                        min(bottom, y0 + grid["span"] + margin),
                    )
                    picture = reader.read(outer, resolution, exact=True)
                    feature_key = f"{settings['scene_fingerprint']}:{outer}:{picture.size}"
                    features = store.object_features(
                        feature_key, picture, self.engine.objects, resolution, colour
                    )
                    classified = self.engine.objects.classify(
                        features,
                        image_vectors,
                        text_vectors,
                        negative_vectors,
                        text_required=bool(settings["instruction"]),
                    )
                    boxes = self.engine.objects.boxes(features, picture.width, picture.height)
                    candidates = []
                    for i in np.flatnonzero(classified["scores"] >= 0.01):
                        job.check()
                        b = boxes[i]
                        sx, sy = (outer[2] - outer[0]) / picture.width, (outer[3] - outer[1]) / picture.height
                        bounds = (
                            round(outer[0] + b[0] * sx),
                            round(outer[1] + b[1] * sy),
                            round(outer[0] + b[2] * sx),
                            round(outer[1] + b[3] * sy),
                        )
                        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
                        if (
                            min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 8
                            or not x0 <= cx < min(right, x0 + grid["span"])
                            or not y0 <= cy < min(bottom, y0 + grid["span"])
                        ):
                            continue
                        if any(
                            intersection_over_union(bounds, box_tuple(e, reader.width, reader.height)) > 0.5
                            for e in exclusions
                        ):
                            continue
                        candidates.append((bounds, float(classified["scores"][i]), int(i)))
                    records = []
                    for bounds, score, i in local_suppress(candidates, 0.4):
                        image_score = (
                            float(classified["image_scores"][i])
                            if classified["image_scores"] is not None
                            else None
                        )
                        text_score = (
                            float(classified["text_scores"][i])
                            if classified["text_scores"] is not None
                            else None
                        )
                        records.append(
                            (
                                scan,
                                tile,
                                bounds[0],
                                bounds[1],
                                bounds[2] - bounds[0],
                                bounds[3] - bounds[1],
                                score,
                                None,
                                text_score,
                                0,
                                image_score,
                                None,
                                image_ids[int(classified["image_indexes"][i])] if image_ids else None,
                                int(classified["support"][i]) if image_ids else 0,
                                None,
                                None,
                                "not_required",
                                0,
                                None,
                                None,
                                None,
                                None,
                            )
                        )
                    job.check()
                    with self.workshop.connect() as db:
                        db.execute("DELETE FROM similar_candidates WHERE scan=? AND tile=?", (scan, tile))
                        db.executemany(
                            "INSERT INTO similar_candidates(scan,tile,x,y,w,h,score,appearance,semantic,angle,detail,structure,matched_reference,support,aerial,geometry,geometry_status,inliers,rotation,scale,quad,adapted) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            records,
                        )
                        db.execute("UPDATE similar_scans SET next_tile=? WHERE id=?", (tile + 1, scan))
                offset += grid["count"]
        settings["neural_cache"] = dict(
            hits=store.object_hits - start_hits, misses=store.object_misses - start_misses
        )
        result = self.similar.finish(job, settings, scan, threshold)
        result["assistant"] = understood
        result["message"] = (
            f"{result['total_results']:,} نتیجهٔ هم‌نوع با مدل تشخیص ذخیره شد. " + understood["answer_fa"]
        )
        return result
