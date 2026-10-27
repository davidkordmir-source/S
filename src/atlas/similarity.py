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
import numpy as np
from PIL import Image

from .config import INDEX_VERSION
from .errors import UserError
from .image_io import read_picture
from .workshop import box_tuple, intersection_over_union, rect_dict

CORE = 768
PAGE = 200
ANGLES = tuple(range(0, 360, 45))


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
                    "INSERT INTO image_references VALUES(?,?,?,?,?,?,?)",
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
            picture = reader.read(box, 1024)
        return self.add_reference(
            picture, "نمونه از تصویر", scene, rect_dict(box), (box[2] - box[0], box[3] - box[1])
        )

    def plan(
        self,
        scene,
        reference,
        width,
        height,
        scope=None,
        broad=False,
        rotate=True,
        intelligent=True,
        **ignored,
    ):
        info = self.workshop.scene(scene)
        self.workshop.scene_path(scene)
        self.reference(reference)
        if not (8 <= width <= 8192 and 8 <= height <= 8192):
            raise UserError("اندازهٔ هدف باید بین ۸ و ۸۱۹۲ پیکسل باشد.")
        box = (
            box_tuple(scope, info["width"], info["height"])
            if scope
            else (0, 0, info["width"], info["height"])
        )
        factor = 1 << max(0, math.floor(math.log2(max(width, height) / 128)))
        span = CORE * factor
        columns = math.ceil((box[2] - box[0]) / span)
        rows = math.ceil((box[3] - box[1]) / span)
        settings = dict(
            method="similar",
            scene=scene,
            reference=reference,
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
            algorithm="reference-v2",
        )
        scan = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with self.workshop.connect() as db:
            row = db.execute("SELECT next_tile FROM similar_scans WHERE id=?", (scan,)).fetchone()
        return dict(
            allowed=True,
            windows=columns * rows,
            completed=row["next_tile"] if row else 0,
            scope=settings["scope"],
            scan=scan,
            settings=settings,
        )

    def search(self, job, threshold=0.8, **options):
        if not math.isfinite(threshold) or not 0.5 <= threshold <= 0.99:
            raise UserError("آستانهٔ شباهت باید بین ۰٫۵ و ۰٫۹۹ باشد.")
        plan = self.plan(**options)
        settings, scan = plan["settings"], plan["scan"]
        reference = self.reference(settings["reference"])
        with Image.open(self.folder / f"{reference['id']}.png") as original:
            sample = original.convert("RGB")
        factor, span, columns = settings["factor"], settings["span"], settings["columns"]
        scales = (0.5, 0.65, 0.8, 1, 1.25, 1.6, 2) if settings["broad"] else (0.8, 1, 1.25)
        templates = []
        for scale in scales:
            tw, th = (
                max(8, round(settings["width"] * scale / factor)),
                max(8, round(settings["height"] * scale / factor)),
            )
            resized = sample.resize((tw, th), Image.Resampling.LANCZOS)
            for angle in ANGLES if settings["rotate"] else (0,):
                rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                template = np.asarray(rotated).copy()
                valid = None
                if angle % 90:
                    mask = Image.new("L", resized.size, 255).rotate(
                        angle, resample=Image.Resampling.BICUBIC, expand=True
                    )
                    valid = (np.asarray(mask) > 250).astype(np.uint8) * 255
                templates.append((template, angle, valid))
        margin = max(max(t.shape[:2]) for t, _, _ in templates) * factor
        box = box_tuple(
            settings["scope"],
            self.workshop.scene(settings["scene"])["width"],
            self.workshop.scene(settings["scene"])["height"],
        )
        templates = [
            (t, a, mask)
            for t, a, mask in templates
            if t.shape[1] * factor <= box[2] - box[0] and t.shape[0] * factor <= box[3] - box[1]
        ]
        if not templates:
            raise UserError("محدوده از نمونه کوچک‌تر است؛ اندازهٔ هدف را کوچک‌تر یا محدوده را بزرگ‌تر کن.")
        reference_vectors = None
        if settings["intelligent"]:
            job.update(1, "آماده‌سازی نمونهٔ تصویری برای مقایسهٔ هوشمند")
            reference_vectors = np.asarray(
                self.workshop.engine.images(
                    [sample.rotate(a, expand=True) for a in (ANGLES if settings["rotate"] else (0,))]
                ),
                dtype=np.float32,
            )
        if reference_vectors is not None and (
            reference_vectors.ndim != 2
            or reference_vectors.shape[1] != 512
            or not np.isfinite(reference_vectors).all()
        ):
            raise UserError("ویژگی‌های نمونهٔ مرجع معتبر نیست؛ سلامت فایل مدل را بررسی کن.")
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
                x0, y0 = box[0] + tile % columns * span, box[1] + tile // columns * span
                x1, y1 = min(box[2], x0 + span + margin), min(box[3], y0 + span + margin)
                picture = reader.read((x0, y0, x1, y1), max(1, math.ceil(max(x1 - x0, y1 - y0) / factor)))
                image = np.asarray(picture)
                # Account for the last partial tile's rounded output dimensions.
                sx, sy = (x1 - x0) / picture.width, (y1 - y0) / picture.height
                candidates = []
                for template, angle, valid in templates:
                    job.check()
                    th, tw = template.shape[:2]
                    if th > image.shape[0] or tw > image.shape[1]:
                        continue
                    constant = np.asarray(template, dtype=np.float32).std(axis=(0, 1)).max() < 3
                    method = cv2.TM_SQDIFF_NORMED if constant else cv2.TM_CCOEFF_NORMED
                    similarity = cv2.matchTemplate(image, template, method, mask=valid)
                    if constant:
                        similarity = 1 - similarity
                    similarity = np.nan_to_num(similarity, nan=-1, posinf=-1, neginf=-1)
                    limit_x = min(similarity.shape[1], max(0, math.ceil(min(span, box[2] - x0) / sx)))
                    limit_y = min(similarity.shape[0], max(0, math.ceil(min(span, box[3] - y0) / sy)))
                    similarity = similarity[:limit_y, :limit_x]
                    if not similarity.size:
                        continue
                    kernel = max(3, min(tw, th) // 2) | 1
                    maxima = cv2.dilate(similarity, np.ones((kernel, kernel), np.uint8))
                    ys, xs = np.nonzero((similarity >= 0.35) & (similarity >= maxima - 1e-7))
                    # Equal-score plateaus become one peak per connected region.
                    mask = np.zeros(similarity.shape, np.uint8)
                    mask[ys, xs] = 1
                    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
                    for label in range(1, n):
                        left, top, width, height = stats[label, :4]
                        region = similarity[top : top + height, left : left + width]
                        ly, lx = np.unravel_index(
                            np.argmax(
                                np.where(labels[top : top + height, left : left + width] == label, region, -2)
                            ),
                            region.shape,
                        )
                        px, py = left + lx, top + ly
                        b = (
                            round(x0 + px * sx),
                            round(y0 + py * sy),
                            min(box[2], round(x0 + (px + tw) * sx)),
                            min(box[3], round(y0 + (py + th) * sy)),
                        )
                        candidates.append((b, float(similarity[py, px]), angle))
                candidates = local_suppress(candidates)
                records = []
                for start in range(0, len(candidates), 16):
                    job.check()
                    chunk = candidates[start : start + 16]
                    semantic = np.ones(len(chunk), np.float32)
                    if reference_vectors is not None:
                        vectors = np.asarray(
                            self.workshop.engine.images([reader.read(b, 384) for b, _, _ in chunk]),
                            np.float32,
                        )
                        if vectors.shape != (len(chunk), 512) or not np.isfinite(vectors).all():
                            raise UserError("مقایسهٔ هوشمند کامل نشد.")
                        semantic = np.clip((vectors @ reference_vectors.T).max(axis=1), 0, 1)
                    for (b, appearance, angle), ai in zip(chunk, semantic, strict=True):
                        score = (
                            0.8 * max(0, appearance) + 0.2 * float(ai)
                            if reference_vectors is not None
                            else max(0, appearance)
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
                                float(ai),
                                angle,
                            )
                        )
                job.check()
                with self.workshop.connect() as db:
                    db.execute("DELETE FROM similar_candidates WHERE scan=? AND tile=?", (scan, tile))
                    db.executemany(
                        "INSERT INTO similar_candidates(scan,tile,x,y,w,h,score,appearance,semantic,angle) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        records,
                    )
                    db.execute("UPDATE similar_scans SET next_tile=? WHERE id=?", (tile + 1, scan))
        job.update(92, "مرتب‌سازی نتایج و حذف هم‌پوشانی؛ تمام نتایج پذیرفته‌شده ذخیره می‌شوند")
        run = uuid.uuid4().hex
        total = 0
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
                        "INSERT INTO similar_detections(run,idx,x,y,w,h,score,appearance,semantic,angle) VALUES(?,?,?,?,?,?,?,?,?,?)",
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
                        ),
                    )
                    db.execute(
                        "INSERT INTO similar_bounds VALUES(?,?,?,?,?)", (result.lastrowid, x, x + w, y, y + h)
                    )
                    total += 1
                job.check()
                settings.update(threshold=threshold, total_results=total, scan=scan)
                db.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                    (run, settings["scene"], None, None, json.dumps(settings), "[]", time.time()),
                )
            result = self.workshop.run(run)
            result.update(
                run=run,
                message=f"جست‌وجو کامل شد؛ {total:,} مورد مشابه ذخیره شد. تمام بخش‌های محدوده بررسی شدند.",
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
            appearance=round(row["appearance"], 5),
            semantic=round(row["semantic"], 5),
            angle=row["angle"],
        )

    def results(self, run, offset=0, limit=PAGE, viewport=None):
        with self.workshop.connect() as db:
            if viewport:
                b = viewport
                rows = db.execute(
                    "SELECT d.* FROM similar_bounds b CROSS JOIN similar_detections d ON d.key=b.key WHERE d.run=? AND b.x1>=? AND b.x0<=? AND b.y1>=? AND b.y0<=? ORDER BY d.score DESC LIMIT ?",
                    (run, b["x"], b["x"] + b["width"], b["y"], b["y"] + b["height"], limit + 1),
                ).fetchall()
                return {
                    "detections": [self.detection(r) for r in rows[:limit]],
                    "truncated": len(rows) > limit,
                }
            rows = db.execute(
                "SELECT * FROM similar_detections WHERE run=? ORDER BY idx LIMIT ? OFFSET ?",
                (run, limit, offset),
            ).fetchall()
        return {"detections": [self.detection(r) for r in rows]}

    def one(self, run, index):
        with self.workshop.connect() as db:
            row = db.execute(
                "SELECT * FROM similar_detections WHERE run=? AND idx=?", (run, index)
            ).fetchone()
        if not row:
            raise UserError("نتیجه پیدا نشد.")
        return self.detection(row)

    def iterate(self, run):
        with self.workshop.connect() as db:
            cursor = db.execute("SELECT * FROM similar_detections WHERE run=? ORDER BY idx", (run,))
            for row in cursor:
                yield self.detection(row)


def local_suppress(candidates):
    """Only suppress highly overlapping proposals, with a bounded spatial grid."""
    chosen = []
    grid = {}
    cell = 256
    for box, score, angle in sorted(candidates, key=lambda item: item[1], reverse=True):
        x0, y0, x1, y1 = box
        keys = [
            (x, y)
            for y in range(y0 // cell, (y1 - 1) // cell + 1)
            for x in range(x0 // cell, (x1 - 1) // cell + 1)
        ]
        neighbours = {i for key in keys for i in grid.get(key, [])}
        if any(intersection_over_union(box, chosen[i][0]) > 0.5 for i in neighbours):
            continue
        i = len(chosen)
        chosen.append((box, score, angle))
        for key in keys:
            grid.setdefault(key, []).append(i)
    return chosen
