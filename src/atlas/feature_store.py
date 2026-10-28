"""Disk-bounded scene feature cache and resumable query-independent scene index."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
import zipfile

import numpy as np

from .config import DETAIL_REVISION
from .errors import UserError
from .visual_features import descriptor, model_picture, shape_picture, unit

VERSION = "scene-features-v1:" + DETAIL_REVISION


class FeatureStore:
    def __init__(self, workshop):
        self.workshop = workshop
        self.folder = workshop.folder / "scene-features"
        self.folder.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.object_hits = 0
        self.object_misses = 0
        with workshop.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scene_feature_cache(key TEXT PRIMARY KEY,path TEXT,bytes INTEGER,used REAL);
                CREATE TABLE IF NOT EXISTS scene_indexes(id TEXT PRIMARY KEY,scene TEXT,settings TEXT,next_tile INTEGER,total_tiles INTEGER,created REAL);
                CREATE TABLE IF NOT EXISTS scene_index_vectors(idx TEXT,tile INTEGER,x INTEGER,y INTEGER,w INTEGER,h INTEGER,vector BLOB,PRIMARY KEY(idx,tile));
                CREATE VIRTUAL TABLE IF NOT EXISTS scene_index_bounds USING rtree(key,x0,x1,y0,y1);
                CREATE TRIGGER IF NOT EXISTS scene_index_bounds_insert AFTER INSERT ON scene_index_vectors BEGIN
                    INSERT OR REPLACE INTO scene_index_bounds VALUES(new.rowid,new.x,new.x+new.w,new.y,new.y+new.h);
                END;
                CREATE TRIGGER IF NOT EXISTS scene_index_bounds_delete AFTER DELETE ON scene_index_vectors BEGIN
                    DELETE FROM scene_index_bounds WHERE key=old.rowid;
                END;
            """)

    def fingerprint(self, scene):
        path = self.workshop.scene_path(scene)
        stat = path.stat()
        return f"{scene}:{stat.st_size}:{stat.st_mtime_ns}:{VERSION}"

    def cached(self, key, picture):
        key = hashlib.sha256(key.encode()).hexdigest()
        path = self.folder / (key + ".npz")
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    grid = arrays["grid"].astype(np.float32)
                    cls = arrays["cls"].astype(np.float32)
                if (
                    grid.shape != (picture.height // 14, picture.width // 14, 384)
                    or cls.shape != (384,)
                    or not np.isfinite(grid).all()
                    or not np.isfinite(cls).all()
                ):
                    raise ValueError("Invalid feature cache")
                self.hits += 1
                with self.workshop.connect() as db:
                    db.execute("UPDATE scene_feature_cache SET used=? WHERE key=?", (time.time(), key))
                return dict(grid=unit(grid), cls=unit(cls))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                path.unlink(missing_ok=True)
        self.misses += 1
        value = self.workshop.engine.details([picture])[0]
        if shutil.disk_usage(self.folder).free < 256 * 1024**2:
            raise UserError("فضای کافی برای کش ویژگی‌های تصویر نیست؛ فضای دیسک را آزاد کن.")
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, grid=value["grid"].astype(np.float16), cls=value["cls"].astype(np.float16)
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        with self.workshop.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO scene_feature_cache VALUES(?,?,?,?)",
                (key, path.name, path.stat().st_size, time.time()),
            )
        self.trim()
        return value

    def trim(self, budget=4 * 1024**3):
        # A cache eviction changes performance only; complete search never skips a region.
        with self.workshop.connect() as db:
            total = db.execute("SELECT COALESCE(SUM(bytes),0) FROM scene_feature_cache").fetchone()[0]
            if total <= budget:
                return
            for row in db.execute("SELECT * FROM scene_feature_cache ORDER BY used").fetchall():
                (self.folder / row["path"]).unlink(missing_ok=True)
                db.execute("DELETE FROM scene_feature_cache WHERE key=?", (row["key"],))
                total -= row["bytes"]
                if total <= budget:
                    break

    def object_features(self, key, picture, detector, resolution=768, colour=False):
        key = (
            "object-"
            + hashlib.sha256(f"{key}:{detector.revision()}:{resolution}:{colour}:v1".encode()).hexdigest()
        )
        path = self.folder / (key + ".npz")
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {
                        k: saved[k].astype(np.float32)
                        for k in ("classes", "boxes", "shift", "scale", "objectness")
                    }
                n = arrays["classes"].shape[0]
                if (
                    not 1 <= n <= 4096
                    or arrays["classes"].shape != (n, 512)
                    or arrays["boxes"].shape != (n, 4)
                    or any(arrays[k].shape != (n,) for k in ("shift", "scale", "objectness"))
                    or any(not np.isfinite(a).all() for a in arrays.values())
                ):
                    raise ValueError("Invalid object feature cache")
                arrays["classes"] = unit(arrays["classes"])
                self.object_hits += 1
                with self.workshop.connect() as db:
                    db.execute("UPDATE scene_feature_cache SET used=? WHERE key=?", (time.time(), key))
                return arrays
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                path.unlink(missing_ok=True)
        self.object_misses += 1
        arrays = detector.scene(picture, resolution, colour)
        if shutil.disk_usage(self.folder).free < 256 * 1024**2:
            raise UserError("فضای دیسک برای ذخیرهٔ ویژگی‌های مدل تشخیص کافی نیست.")
        temporary = self.folder / f"{key}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **{k: v.astype(np.float16) for k, v in arrays.items()})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        with self.workshop.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO scene_feature_cache VALUES(?,?,?,?)",
                (key, path.name, path.stat().st_size, time.time()),
            )
        self.trim()
        return arrays

    def index_plan(self, scene, block=512):
        if block not in (256, 512, 1024):
            raise UserError("اندازهٔ نمایه معتبر نیست.")
        info = self.workshop.scene(scene)
        fingerprint = self.fingerprint(scene)
        stride = block // 2
        columns = math.ceil(info["width"] / stride)
        rows = math.ceil(info["height"] / stride)
        settings = dict(scene=scene, block=block, stride=stride, columns=columns, fingerprint=fingerprint)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with self.workshop.connect() as db:
            row = db.execute("SELECT next_tile FROM scene_indexes WHERE id=?", (key,)).fetchone()
        # Vectors are384 float16 values plus conservative SQLite/coordinate overhead.
        return dict(
            id=key,
            settings=settings,
            windows=columns * rows,
            completed=row["next_tile"] if row else 0,
            estimated_bytes=columns * rows * 1400,
        )

    def build(self, job, scene, block=512, budget_gb=20):
        plan = self.index_plan(scene, block)
        key = plan["id"]
        settings = plan["settings"]
        limit = int(budget_gb * 1024**3)
        if plan["estimated_bytes"] > limit:
            raise UserError(
                "حجم برآوردشدهٔ نمایه از بودجهٔ انتخابی بیشتر است؛ اندازهٔ نمایه را بزرگ‌تر یا بودجهٔ دیسک را افزایش بده."
            )
        with self.workshop.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO scene_indexes VALUES(?,?,?,?,?,?)",
                (key, scene, json.dumps(settings), 0, plan["windows"], time.time()),
            )
        with self.workshop.reader(scene) as reader:
            for tile in range(plan["completed"], plan["windows"]):
                job.update(
                    tile / plan["windows"] * 95,
                    f"نمایه‌سازی تصویر: {tile + 1:,} از {plan['windows']:,}؛ پیشرفت ذخیره می‌شود",
                )
                if shutil.disk_usage(self.folder).free < 256 * 1024**2:
                    raise UserError("فضای کافی برای ادامهٔ نمایه‌سازی نیست؛ پیشرفت قبلی ذخیره شده است.")
                x = (tile % settings["columns"]) * settings["stride"]
                y = (tile // settings["columns"]) * settings["stride"]
                x1 = min(reader.width, x + block)
                y1 = min(reader.height, y + block)
                picture = shape_picture(reader.read((x, y, x1, y1), 224, exact=True))
                prepared = model_picture(picture)
                features = self.workshop.engine.details([prepared])[0]
                vector = unit(descriptor(features)["pooled"]).astype(np.float16).tobytes()
                job.check()
                with self.workshop.connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO scene_index_vectors VALUES(?,?,?,?,?,?,?)",
                        (key, tile, x, y, x1 - x, y1 - y, vector),
                    )
                    db.execute("UPDATE scene_indexes SET next_tile=? WHERE id=?", (tile + 1, key))
        return dict(
            message="نمایهٔ آفلاین تصویر آماده شد؛ جست‌وجوی سریع قابل استفاده است.",
            index=self.index_plan(scene, block),
        )

    def shortlist(self, scene, references, block=512, top=1000, scope=None):
        import heapq

        plan = self.index_plan(scene, block)
        if plan["completed"] != plan["windows"]:
            raise UserError(
                "برای حالت سریع، ابتدا نمایهٔ کامل این تصویر را آماده کن؛ یا حالت بررسی کامل را انتخاب کن."
            )
        vectors = np.asarray(references, np.float32)
        heap = []
        with self.workshop.connect() as db:
            query = "SELECT v.* FROM scene_index_vectors v"
            parameters = [plan["id"]]
            if scope:
                query += " JOIN scene_index_bounds b ON b.key=v.rowid WHERE v.idx=? AND b.x0<? AND b.x1>? AND b.y0<? AND b.y1>?"
                parameters += [
                    scope["x"] + scope["width"],
                    scope["x"],
                    scope["y"] + scope["height"],
                    scope["y"],
                ]
            else:
                query += " WHERE v.idx=?"
            cursor = db.execute(query + " ORDER BY v.tile", parameters)
            while rows := cursor.fetchmany(512):
                matrix = np.stack([np.frombuffer(r["vector"], np.float16).astype(np.float32) for r in rows])
                scores = np.einsum("nc,rc->nr", unit(matrix), vectors, optimize=False).max(axis=1)
                for row, score in zip(rows, scores, strict=True):
                    value = (float(score), row["tile"], row["x"], row["y"], row["w"], row["h"])
                    if len(heap) < top:
                        heapq.heappush(heap, value)
                    elif value > heap[0]:
                        heapq.heapreplace(heap, value)
        return [
            dict(x=r[2], y=r[3], width=r[4], height=r[5], retrieval_score=r[0])
            for r in sorted(heap, reverse=True)
        ]

    def overview(self, scene):
        return [self.index_plan(scene, size) for size in (256, 512, 1024)]
