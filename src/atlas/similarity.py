"""Reference-image search: complete tiled coverage, resumable disk candidates, paged results."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from pathlib import Path

import cv2
from PIL import Image

from .config import DETAIL_REVISION, INDEX_VERSION
from .errors import UserError
from .feature_store import FeatureStore
from .image_io import read_picture
from .reference_library import ReferenceLibrary
from .reference_matcher import ReferenceMatcher
from .segmentation import Segmentation
from .target_learning import TargetLearning
from .visual_features import shape_picture
from .workshop import box_tuple, intersection_over_union, rect_dict

CORE = 768
PAGE = 200
ANGLES = tuple(range(0, 360, 45))
LEVELS = (("high", 0.85, "بسیار مشابه"), ("medium", 0.7, "مشابه"), ("review", 0, "نیازمند بررسی"))


def similarity_level(score):
    return next(name for name, minimum, _ in LEVELS if score >= minimum)


class Similarity:
    def __init__(self, workshop):
        self.workshop = workshop
        self.folder = workshop.folder / "references"
        self.folder.mkdir(exist_ok=True)
        cv2.setNumThreads(2)
        with workshop.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS image_references(id TEXT PRIMARY KEY,name TEXT,scene TEXT,bbox TEXT,width INTEGER,height INTEGER,created REAL);
                CREATE TABLE IF NOT EXISTS similar_scans(id TEXT PRIMARY KEY,scene TEXT,reference TEXT REFERENCES image_references(id) ON DELETE CASCADE,settings TEXT,next_tile INTEGER,total_tiles INTEGER,created REAL);
                CREATE TABLE IF NOT EXISTS similar_candidates(id INTEGER PRIMARY KEY,scan TEXT REFERENCES similar_scans(id) ON DELETE CASCADE,tile INTEGER,x INTEGER,y INTEGER,w INTEGER,h INTEGER,score REAL,appearance REAL,semantic REAL,angle INTEGER);
                CREATE INDEX IF NOT EXISTS candidate_score ON similar_candidates(scan,score DESC);
                CREATE TABLE IF NOT EXISTS similar_detections(key INTEGER PRIMARY KEY,run TEXT,idx INTEGER,x INTEGER,y INTEGER,w INTEGER,h INTEGER,score REAL,appearance REAL,semantic REAL,angle INTEGER,UNIQUE(run,idx));
                CREATE VIRTUAL TABLE IF NOT EXISTS similar_bounds USING rtree(key,x0,x1,y0,y1);
            """)
            for table in ("similar_candidates", "similar_detections"):
                existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                for column in ("detail", "structure"):
                    if column not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} REAL")
                for column, kind in (("matched_reference", "TEXT"), ("support", "INTEGER")):
                    if column not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        self.library = ReferenceLibrary(self)
        self.segmentation = Segmentation(self)
        self.learning = TargetLearning(self)
        self.feature_store = FeatureStore(workshop)
        with workshop.connect() as db:
            for table in ("similar_candidates", "similar_detections"):
                columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in [
                    ("aerial", "REAL"),
                    ("geometry", "REAL"),
                    ("geometry_status", "TEXT"),
                    ("inliers", "INTEGER"),
                    ("rotation", "REAL"),
                    ("scale", "REAL"),
                    ("quad", "TEXT"),
                    ("adapted", "REAL"),
                ]:
                    if name not in columns:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def references(self):
        with self.workshop.connect() as db:
            return [
                self._reference(dict(row))
                for row in db.execute("SELECT * FROM image_references ORDER BY created DESC")
            ]

    @staticmethod
    def _reference(row):
        row["bbox"] = json.loads(row["bbox"]) if row["bbox"] else None
        row["image_url"] = f"/api/workspace/references/{row['id']}/image"
        row["source_image_url"] = (
            f"/api/workspace/references/{row.get('source_reference') or row['id']}/image"
        )
        row["has_mask"] = bool(row.get("source_reference"))
        return row

    def reference(self, id):
        with self.workshop.connect() as db:
            row = db.execute("SELECT * FROM image_references WHERE id=?", (id,)).fetchone()
        if not row:
            raise UserError("نمونهٔ مرجع پیدا نشد.")
        return self._reference(dict(row))

    def add_reference(self, picture, name, scene=None, bbox=None, original_size=None):
        if min(picture.size) < 8:
            raise UserError("نمونه باید حداقل ۸ در ۸ پیکسل باشد.")
        identifier = uuid.uuid4().hex
        original = original_size or picture.size
        picture = picture.convert("RGB")
        picture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        path = self.folder / f"{identifier}.png"
        picture.save(path)
        try:
            with self.workshop.connect() as db:
                db.execute(
                    "INSERT INTO image_references(id,name,scene,bbox,width,height,created) VALUES(?,?,?,?,?,?,?)",
                    (
                        identifier,
                        Path(name.replace("\\", "/")).name[:160],
                        scene,
                        json.dumps(bbox) if bbox else None,
                        *original,
                        time.time(),
                    ),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return self.reference(identifier)

    def add_file(self, data, name):
        return self.add_reference(read_picture(data), name)

    def add_box(self, scene, bbox):
        info = self.workshop.scene(scene)
        box = box_tuple(bbox, info["width"], info["height"])
        with self.workshop.reader(scene) as reader:
            picture = reader.read(box, 1024, exact=True)
        return self.add_reference(
            picture, "نمونه از تصویر", scene, rect_dict(box), (box[2] - box[0], box[3] - box[1])
        )

    def plan(
        self,
        scene,
        reference=None,
        width=128,
        height=128,
        scope=None,
        broad=False,
        rotate=True,
        intelligent=True,
        group=None,
        aerial=True,
        geometry=True,
        strict_geometry=False,
        fine_pose=True,
        adapt=True,
        mode="complete",
        index_block=512,
        fast_regions=1000,
        match_mode="exact",
        **ignored,
    ):
        if match_mode == "instances":
            from .semantic_search import SemanticSearch

            return SemanticSearch(self).plan(
                scene=scene, reference=reference, group=group, scope=scope, mode=mode, **ignored
            )
        if match_mode != "exact":
            raise UserError("نوع تطابق معتبر نیست.")
        info = self.workshop.scene(scene)
        if group and self.library.group(group)["archived"]:
            raise UserError("هدف حذف شده است؛ یک هدف فعال را انتخاب کن.")
        self.workshop.scene_path(scene)
        references = (
            [
                r
                for r in self.library.group(group)["references"]
                if r["enabled"] and r["label"] == 1 and r["partition"] == "train"
            ]
            if group
            else [self.reference(reference)]
            if reference
            else []
        )
        if not references:
            raise UserError("حداقل یک نمونهٔ فعال برای هدف اضافه کن؛ چند نمونه اجباری نیست.")
        reference = references[0]["id"]
        if mode not in ("complete", "fast"):
            raise UserError("حالت جست‌وجو معتبر نیست.")
        if strict_geometry and not geometry:
            raise UserError("برای حالت سخت‌گیرانه، بررسی هندسی را فعال کن.")
        if not intelligent and (mode == "fast" or adapt and group and self.learning.status(group)["ready"]):
            raise UserError("حالت سریع و آموزش اختصاصی به جست‌وجوی هوشمند نیاز دارند.")
        negative = (
            [
                dict(id=r["id"], name=r["name"])
                for r in self.library.group(group)["references"]
                if r["enabled"] and r["label"] == 0 and r["partition"] == "train"
            ]
            if group
            else []
        )
        learning = self.learning.status(group) if group else None
        index = self.feature_store.index_plan(scene, index_block) if mode == "fast" else None
        if index and index["completed"] != index["windows"]:
            raise UserError(
                "نمایهٔ تصویر هنوز کامل نیست؛ ابتدا نمایه‌سازی را تمام کن یا بررسی کامل را انتخاب کن."
            )
        if not (8 <= width <= 8192 and 8 <= height <= 8192):
            raise UserError("اندازهٔ هدف باید بین ۸ و ۸۱۹۲ پیکسل باشد.")
        box = (
            box_tuple(scope, info["width"], info["height"])
            if scope
            else (0, 0, info["width"], info["height"])
        )
        factor = 1 << max(0, math.floor(math.log2(max(width, height) / 128)))
        views = [
            dict(
                id=r["id"],
                name=r["name"],
                width=width
                if i == 0
                else max(8, round(max(width, height) * r["width"] / max(r["width"], r["height"]))),
                height=height
                if i == 0
                else max(8, round(max(width, height) * r["height"] / max(r["width"], r["height"]))),
            )
            for i, r in enumerate(references)
        ]
        span = CORE * factor
        columns = math.ceil((box[2] - box[0]) / span)
        rows = math.ceil((box[3] - box[1]) / span)
        settings = dict(
            method="similar",
            scene=scene,
            reference=reference,
            group=group,
            references=views,
            reference_count=len(views),
            colour_evidence=False,
            scene_fingerprint=self.feature_store.fingerprint(scene),
            negative_references=negative,
            training_revision=learning["revision"] if learning else None,
            adapter=learning["model"] if adapt and learning and learning["ready"] else None,
            aerial=aerial,
            geometry=geometry,
            strict_geometry=strict_geometry,
            fine_pose=fine_pose,
            adapt=adapt,
            mode=mode,
            index_block=index_block,
            index_id=index["id"] if index else None,
            fast_regions=fast_regions,
            width=width,
            height=height,
            scope=rect_dict(box),
            broad=broad,
            rotate=rotate,
            intelligent=intelligent,
            factor=factor,
            span=span,
            columns=columns,
            windows=columns * rows,
            feature=INDEX_VERSION,
            algorithm="reference-v7-nonconstant:" + DETAIL_REVISION[:12],
        )
        scan = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with self.workshop.connect() as db:
            row = db.execute("SELECT next_tile,total_tiles FROM similar_scans WHERE id=?", (scan,)).fetchone()
        return dict(
            allowed=True,
            windows=row["total_tiles"] if row else columns * rows,
            completed=row["next_tile"] if row else 0,
            scope=settings["scope"],
            scan=scan,
            settings=settings,
        )

    def search(self, job, threshold=None, **options):
        if options.get("match_mode") == "instances":
            from .semantic_search import SemanticSearch

            return SemanticSearch(self).search(
                job, threshold=0.12 if threshold is None else threshold, **options
            )
        threshold = 0.55 if threshold is None else threshold
        if not math.isfinite(threshold) or not 0.4 <= threshold <= 0.99:
            raise UserError("آستانهٔ شباهت باید بین ۰٫۴ و ۰٫۹۹ باشد.")
        group = options.get("group")
        if group and options.get("adapt", True) and options.get("intelligent", True):
            state = self.learning.status(group)
            if not state["ready"] and state["positive"] and state["negative"]:
                self.learning.train(job, group)
        plan = self.plan(**options)
        settings, scan = plan["settings"], plan["scan"]
        reference = self.reference(settings["reference"])
        matcher = ReferenceMatcher(self, settings, job)
        selected = None
        if settings["mode"] == "fast":
            with self.workshop.connect() as db:
                saved = db.execute("SELECT settings FROM similar_scans WHERE id=?", (scan,)).fetchone()
            if saved:
                selected = json.loads(saved["settings"]).get("selected_tiles")
            if selected is None:
                areas = self.feature_store.shortlist(
                    settings["scene"],
                    [vector for bank in matcher.bank for vector in bank["pooled"]],
                    settings["index_block"],
                    settings["fast_regions"],
                    scope=settings["scope"],
                )
                scope = settings["scope"]
                span = settings["span"]
                selected_set = set()
                for area in areas:
                    left = max(scope["x"], area["x"])
                    top = max(scope["y"], area["y"])
                    right = min(scope["x"] + scope["width"], area["x"] + area["width"])
                    bottom = min(scope["y"] + scope["height"], area["y"] + area["height"])
                    if left >= right or top >= bottom:
                        continue
                    for ty in range(
                        int((top - scope["y"]) // span), int((bottom - 1 - scope["y"]) // span) + 1
                    ):
                        for tx in range(
                            int((left - scope["x"]) // span), int((right - 1 - scope["x"]) // span) + 1
                        ):
                            selected_set.add(ty * settings["columns"] + tx)
                selected = sorted(selected_set)
            settings["selected_tiles"] = selected
            settings["full_windows"] = settings["windows"]
            settings["windows"] = len(selected)
            plan["windows"] = len(selected)
        factor, span, columns = settings["factor"], settings["span"], settings["columns"]
        margin = math.ceil(
            max(max(v["width"], v["height"]) for v in settings["references"])
            * max(max(matcher.scales), (3 if settings["broad"] else 1.6) if settings["fine_pose"] else 0)
            * (2**0.5 if settings["rotate"] else 1)
        )
        info = self.workshop.scene(settings["scene"])
        box = box_tuple(settings["scope"], info["width"], info["height"])
        if not any(
            v["width"] * min(matcher.scales) <= box[2] - box[0]
            and v["height"] * min(matcher.scales) <= box[3] - box[1]
            for v in settings["references"]
        ):
            raise UserError("محدوده از نمونه کوچک‌تر است؛ اندازهٔ هدف را کوچک‌تر یا محدوده را بزرگ‌تر کن.")
        with self.workshop.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO similar_scans VALUES(?,?,?,?,?,?,?)",
                (
                    scan,
                    settings["scene"],
                    reference["id"],
                    json.dumps(settings),
                    0,
                    plan["windows"],
                    time.time(),
                ),
            )
        with self.workshop.reader(settings["scene"]) as reader:
            for tile in range(plan["completed"], plan["windows"]):
                job.update(
                    2 + tile / plan["windows"] * 88,
                    f"جست‌وجوی مشابه: بخش {tile + 1:,} از {plan['windows']:,}؛ ادامهٔ خودکار از آخرین بخش ذخیره‌شده",
                )
                if shutil.disk_usage(self.folder).free < 256 * 1024 * 1024:
                    raise UserError(
                        "فضای دیسک برای ذخیرهٔ جست‌وجو کافی نیست؛ پیشرفت تا بخش قبلی ذخیره شده است."
                    )
                position = selected[tile] if selected is not None else tile
                x0, y0 = box[0] + position % columns * span, box[1] + position // columns * span
                x1, y1 = min(box[2], x0 + span + margin), min(box[3], y0 + span + margin)
                picture = reader.read(
                    (x0, y0, x1, y1), max(1, math.ceil(max(x1 - x0, y1 - y0) / factor)), exact=True
                )
                sx, sy = (x1 - x0) / picture.width, (y1 - y0) / picture.height
                candidates = matcher.proposals(
                    shape_picture(picture), sx, sy, span, x0, y0, box, local_suppress
                )
                records = []
                for start in range(0, len(candidates), 16):
                    job.check()
                    chunk = candidates[start : start + 16]
                    crops = [reader.read(item[0], 384, exact=True) for item in chunk]
                    records.extend(matcher.score(chunk, crops, scan, tile))
                job.check()
                with self.workshop.connect() as db:
                    db.execute("DELETE FROM similar_candidates WHERE scan=? AND tile=?", (scan, tile))
                    db.executemany(
                        "INSERT INTO similar_candidates(scan,tile,x,y,w,h,score,appearance,semantic,angle,detail,structure,matched_reference,support,aerial,geometry,geometry_status,inliers,rotation,scale,quad,adapted) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        records,
                    )
                    db.execute("UPDATE similar_scans SET next_tile=? WHERE id=?", (tile + 1, scan))
        return self.finish(job, settings, scan, threshold)

    def finish(self, job, settings, scan, threshold):
        job.update(92, "مرتب‌سازی نتایج و حذف هم‌پوشانی؛ تمام نتایج پذیرفته‌شده ذخیره می‌شوند")
        run = uuid.uuid4().hex
        total = 0
        level_counts = dict(high=0, medium=0, review=0)
        try:
            with self.workshop.connect() as read_db, self.workshop.connect() as db:
                cursor = read_db.execute(
                    "SELECT * FROM similar_candidates WHERE scan=? AND score>=? ORDER BY score DESC,id",
                    (scan, threshold),
                )
                for number, row in enumerate(cursor):
                    if number % 64 == 0:
                        job.check()
                    x, y, w, h = (row[k] for k in ("x", "y", "w", "h"))
                    neighbours = db.execute(
                        "SELECT d.* FROM similar_bounds b CROSS JOIN similar_detections d ON d.key=b.key WHERE d.run=? AND b.x1>? AND b.x0<? AND b.y1>? AND b.y0<?",
                        (run, x, x + w, y, y + h),
                    ).fetchall()
                    if any(
                        intersection_over_union(
                            (x, y, x + w, y + h), (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])
                        )
                        > 0.3
                        for r in neighbours
                    ):
                        continue
                    result = db.execute(
                        "INSERT INTO similar_detections(run,idx,x,y,w,h,score,appearance,semantic,angle,detail,structure,matched_reference,support,aerial,geometry,geometry_status,inliers,rotation,scale,quad,adapted) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run,
                            total,
                            x,
                            y,
                            w,
                            h,
                            row["score"],
                            row["appearance"],
                            row["semantic"],
                            row["angle"],
                            row["detail"],
                            row["structure"],
                            row["matched_reference"],
                            row["support"],
                            *[
                                row[k]
                                for k in (
                                    "aerial",
                                    "geometry",
                                    "geometry_status",
                                    "inliers",
                                    "rotation",
                                    "scale",
                                    "quad",
                                    "adapted",
                                )
                            ],
                        ),
                    )
                    db.execute(
                        "INSERT INTO similar_bounds VALUES(?,?,?,?,?)", (result.lastrowid, x, x + w, y, y + h)
                    )
                    total += 1
                    level_counts[similarity_level(row["score"])] += 1
                job.check()
                settings.update(
                    threshold=threshold, total_results=total, scan=scan, level_counts=level_counts
                )
                db.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                    (run, settings["scene"], None, None, json.dumps(settings), "[]", time.time()),
                )
            result = self.workshop.run(run)
            result.update(
                run=run,
                message=f"جست‌وجو کامل شد؛ {total:,} مورد مشابه ذخیره شد. "
                + (
                    "بخش‌های برگزیدهٔ نمایه بررسی شدند؛ حالت سریع ممکن است مواردی را جا بیندازد."
                    if settings["mode"] == "fast"
                    else "تمام بخش‌های محدوده بررسی شدند."
                ),
            )
            return result
        except Exception:
            with self.workshop.connect() as db:
                db.execute(
                    "DELETE FROM similar_bounds WHERE key IN (SELECT key FROM similar_detections WHERE run=?)",
                    (run,),
                )
                db.execute("DELETE FROM similar_detections WHERE run=?", (run,))
            raise

    @staticmethod
    def detection(row):
        return dict(
            index=row["idx"],
            bbox=dict(x=row["x"], y=row["y"], width=row["w"], height=row["h"]),
            score=round(row["score"], 5),
            appearance=round(row["appearance"], 5) if row["appearance"] is not None else None,
            semantic=round(row["semantic"], 5) if row["semantic"] is not None else None,
            angle=row["angle"],
            matched_reference=row["matched_reference"],
            support=row["support"],
            aerial=row["aerial"],
            geometry=row["geometry"],
            geometry_status=row["geometry_status"],
            inliers=row["inliers"],
            rotation=row["rotation"],
            scale=row["scale"],
            quad=json.loads(row["quad"]) if row["quad"] else None,
            adapted=row["adapted"],
            detail=round(row["detail"], 5) if row["detail"] is not None else None,
            structure=round(row["structure"], 5) if row["structure"] is not None else None,
            level=similarity_level(row["score"]),
            similarity_percent=round(row["score"] * 100, 1),
        )

    def results(self, run, offset=0, limit=PAGE, viewport=None, level="all", counts=None):
        ranges = {"all": (0.0, 2.0), "high": (0.85, 2.0), "medium": (0.7, 0.85), "review": (0.0, 0.7)}
        if level not in ranges:
            raise UserError("سطح شباهت معتبر نیست.")
        minimum, maximum = ranges[level]
        with self.workshop.connect() as db:
            if viewport:
                b = viewport
                rows = db.execute(
                    "SELECT d.* FROM similar_bounds b CROSS JOIN similar_detections d ON d.key=b.key WHERE d.run=? AND d.score>=? AND d.score<? AND b.x1>=? AND b.x0<=? AND b.y1>=? AND b.y0<=? ORDER BY d.score DESC LIMIT ?",
                    (
                        run,
                        minimum,
                        maximum,
                        b["x"],
                        b["x"] + b["width"],
                        b["y"],
                        b["y"] + b["height"],
                        limit + 1,
                    ),
                ).fetchall()
                return {
                    "detections": self.annotate_feedback(run, [self.detection(r) for r in rows[:limit]]),
                    "truncated": len(rows) > limit,
                }
            rows = db.execute(
                "SELECT * FROM similar_detections WHERE run=? AND score>=? AND score<? ORDER BY idx LIMIT ? OFFSET ?",
                (run, minimum, maximum, limit, offset),
            ).fetchall()
            if counts is None:  # compatibility with runs saved before levels existed
                counts = dict(high=0, medium=0, review=0)
                for row in db.execute(
                    "SELECT CASE WHEN score>=.85 THEN 'high' WHEN score>=.7 THEN 'medium' ELSE 'review' END AS level,COUNT(*) AS total FROM similar_detections WHERE run=? GROUP BY level",
                    (run,),
                ):
                    counts[row["level"]] = row["total"]
        return {
            "detections": self.annotate_feedback(run, [self.detection(r) for r in rows]),
            "level": level,
            "level_counts": counts,
            "filtered_results": sum(counts.values()) if level == "all" else counts[level],
        }

    def annotate_feedback(self, run, detections):
        with self.workshop.connect() as db:
            labels = {
                r["idx"]: r["label"]
                for r in db.execute("SELECT idx,label FROM reference_feedback WHERE run=?", (run,))
            }
        return [{**d, "feedback": labels.get(d["index"])} for d in detections]

    def one(self, run, index):
        with self.workshop.connect() as db:
            row = db.execute(
                "SELECT * FROM similar_detections WHERE run=? AND idx=?", (run, index)
            ).fetchone()
        if not row:
            raise UserError("نتیجه پیدا نشد.")
        return self.detection(row)

    def page_for(self, run, index, level, limit):
        # Indices are global ranks; a filtered page has its own offset.
        self.one(run, index)
        allowed = {"all": (0.0, 2.0), "high": (0.85, 2.0), "medium": (0.7, 0.85), "review": (0.0, 0.7)}
        if level not in allowed:
            raise UserError("سطح شباهت معتبر نیست.")
        low, high = allowed[level]
        with self.workshop.connect() as db:
            before = db.execute(
                "SELECT COUNT(*) FROM similar_detections WHERE run=? AND idx<? AND score>=? AND score<?",
                (run, index, low, high),
            ).fetchone()[0]
        return before // limit * limit

    def iterate(self, run):
        with self.workshop.connect() as db:
            cursor = db.execute("SELECT * FROM similar_detections WHERE run=? ORDER BY idx", (run,))
            for row in cursor:
                yield self.detection(row)


def local_suppress(candidates, overlap=0.5):
    """Only suppress highly overlapping proposals, with a bounded spatial grid."""
    chosen = []
    grid = {}
    cell = 256
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        box = candidate[0]
        x0, y0, x1, y1 = box
        keys = [
            (x, y)
            for y in range(y0 // cell, (y1 - 1) // cell + 1)
            for x in range(x0 // cell, (x1 - 1) // cell + 1)
        ]
        neighbours = {i for key in keys for i in grid.get(key, [])}
        if any(intersection_over_union(box, chosen[i][0]) > overlap for i in neighbours):
            continue
        i = len(chosen)
        chosen.append(candidate)
        for key in keys:
            grid.setdefault(key, []).append(i)
    return chosen
