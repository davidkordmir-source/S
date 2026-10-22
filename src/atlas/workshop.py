from __future__ import annotations

import hashlib
import io
import json
import math
import sqlite3
import time
import uuid
import warnings
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .config import INDEX_VERSION
from .engine import Engine
from .errors import UserError
from .jobs import Job

MAX_SCENE_BYTES = 150 * 1024 * 1024
MAX_PIXELS = 64_000_000
MAX_WINDOWS = 5000
MAX_EXAMPLES = 500


def read_picture(data: bytes, *, scene: bool = False) -> Image.Image:
    if not data or len(data) > (MAX_SCENE_BYTES if scene else 15 * 1024 * 1024):
        raise UserError("حجم تصویر بیش از حد مجاز است.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if (
                    image.width * image.height > (MAX_PIXELS if scene else 16_000_000)
                    or max(image.size) > 16000
                ):
                    raise UserError(
                        "تصویر بیش از حد بزرگ است. برای تصویر اصلی حداکثر ۶۴ میلیون پیکسل مجاز است."
                    )
                if getattr(image, "n_frames", 1) > 1 or image.mode not in ("RGB", "RGBA", "L", "P"):
                    raise UserError(
                        "یک تصویر RGB معمولی وارد کنید؛ TIFF چندصفحه‌ای، چندباندی و ۱۶بیتی پشتیبانی نمی‌شود."
                    )
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise UserError("تصویر قابل خواندن نیست. از PNG، JPG یا TIFF با رنگ RGB استفاده کنید.") from exc


def box_tuple(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        numbers = [float(box[key]) for key in ("x", "y", "width", "height")]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError()
        x, y, w, h = [round(value) for value in numbers]
        if x < 0 or y < 0 or w < 8 or h < 8 or x + w > width or y + h > height:
            raise ValueError()
        return x, y, x + w, y + h
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UserError("کادر باید داخل تصویر و حداقل ۸ در ۸ پیکسل باشد.") from exc


def rect_dict(box: tuple[int, int, int, int]) -> dict:
    return {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]}


class Workshop:
    def __init__(self, root: Path, engine: Engine):
        self.root, self.engine = root, engine
        self.folder = root / "workshop"
        self.images, self.crops, self.cache = [
            self.folder / name for name in ("images", "examples", "features")
        ]
        for path in (self.images, self.crops, self.cache):
            path.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / "workspace.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS scenes(
                  id TEXT PRIMARY KEY,name TEXT NOT NULL,width INTEGER,height INTEGER,created REAL);
                CREATE TABLE IF NOT EXISTS concepts(
                  id TEXT PRIMARY KEY,name TEXT NOT NULL,revision INTEGER DEFAULT 0,
                  trained_revision INTEGER,model TEXT,created REAL);
                CREATE TABLE IF NOT EXISTS examples(
                  id TEXT PRIMARY KEY,concept TEXT REFERENCES concepts(id) ON DELETE CASCADE,
                  scene TEXT,bbox TEXT,label INTEGER,width INTEGER,height INTEGER,
                  vectors BLOB,feature_version TEXT,created REAL);
                CREATE TABLE IF NOT EXISTS runs(
                  id TEXT PRIMARY KEY,scene TEXT,concept TEXT,revision INTEGER,
                  settings TEXT,results TEXT,created REAL);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def scene(self, id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM scenes WHERE id=?", (id,)).fetchone()
        if row is None:
            raise UserError("تصویر پیدا نشد.")
        return {**dict(row), "preview_url": f"/api/workspace/scenes/{id}/preview"}

    def scene_path(self, id: str) -> Path:
        self.scene(id)
        return self.images / f"{id}.png"

    def scenes(self) -> list[dict]:
        with self.connect() as db:
            return [
                {**dict(row), "preview_url": f"/api/workspace/scenes/{row['id']}/preview"}
                for row in db.execute("SELECT * FROM scenes ORDER BY created DESC")
            ]

    def import_scene(self, data: bytes, name: str, job: Job) -> dict:
        job.update(5, "خواندن تصویر ماهواره‌ای از فایل شما")
        image = read_picture(data, scene=True)
        identifier = uuid.uuid4().hex
        job.update(35, "ذخیرهٔ تصویر روی همین کامپیوتر")
        image.save(self.images / f"{identifier}.png", format="PNG")
        preview = image.copy()
        preview.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        preview.save(self.images / f"{identifier}-preview.jpg", quality=90)
        with self.connect() as db:
            db.execute(
                "INSERT INTO scenes VALUES(?,?,?,?,?)",
                (
                    identifier,
                    Path(name.replace("\\", "/")).name[:160],
                    image.width,
                    image.height,
                    time.time(),
                ),
            )
        return {
            "scene": self.scene(identifier),
            "message": "تصویر شما آماده است. حالا نمونه‌های هدف را مشخص کنید.",
        }

    def concepts(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT c.*,sum(CASE WHEN e.label=1 THEN 1 ELSE 0 END) positive,
                sum(CASE WHEN e.label=0 THEN 1 ELSE 0 END) negative
                FROM concepts c LEFT JOIN examples e ON e.concept=c.id
                GROUP BY c.id ORDER BY c.created DESC""").fetchall()
        result = []
        for row in rows:
            model = json.loads(row["model"]) if row["model"] else None
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "revision": row["revision"],
                    "trained_revision": row["trained_revision"],
                    "positive": row["positive"] or 0,
                    "negative": row["negative"] or 0,
                    "ready": bool(
                        model
                        and row["trained_revision"] == row["revision"]
                        and model.get("feature_version") == INDEX_VERSION
                    ),
                    "window_width": model["window_width"] if model else self.example_size(row["id"])[0],
                    "window_height": model["window_height"] if model else self.example_size(row["id"])[1],
                }
            )
        return result

    def concept(self, id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM concepts WHERE id=?", (id,)).fetchone()
        if row is None:
            raise UserError("موضوع آموزشی پیدا نشد.")
        return dict(row)

    def new_concept(self, name: str) -> dict:
        name = name.strip()
        if not 1 <= len(name) <= 100:
            raise UserError("نام هدف باید بین ۱ تا ۱۰۰ نویسه باشد.")
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO concepts(id,name,created) VALUES(?,?,?)", (identifier, name, time.time()))
        return self.concept(identifier)

    def examples(self, concept: str) -> list[dict]:
        self.concept(concept)
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,scene,bbox,label,width,height FROM examples WHERE concept=? ORDER BY created DESC",
                (concept,),
            ).fetchall()
        return [
            {
                **dict(row),
                "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
                "image_url": f"/api/workspace/examples/{row['id']}/image",
            }
            for row in rows
        ]

    def example_path(self, id: str) -> Path:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM examples WHERE id=?", (id,)).fetchone():
                raise UserError("نمونه پیدا نشد.")
        return self.crops / f"{id}.png"

    def _add_example(
        self,
        concept: str,
        picture: Image.Image,
        label: int,
        scene: str | None = None,
        bbox: dict | None = None,
    ) -> dict:
        self.concept(concept)
        if label not in (0, 1):
            raise UserError("نوع نمونه معتبر نیست.")
        if len(self.examples(concept)) >= MAX_EXAMPLES:
            raise UserError("برای هر موضوع حداکثر ۵۰۰ نمونه مجاز است.")
        identifier = uuid.uuid4().hex
        original_size = picture.size
        picture = picture.copy()
        picture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        picture.save(self.crops / f"{identifier}.png", format="PNG")
        with self.connect() as db:
            db.execute(
                "INSERT INTO examples(id,concept,scene,bbox,label,width,height,created) VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    concept,
                    scene,
                    json.dumps(bbox) if bbox else None,
                    label,
                    *original_size,
                    time.time(),
                ),
            )
            db.execute("UPDATE concepts SET revision=revision+1 WHERE id=?", (concept,))
        return {"id": identifier, "label": label, "image_url": f"/api/workspace/examples/{identifier}/image"}

    def add_box(self, concept: str, scene: str, bbox: dict, label: int) -> dict:
        information = self.scene(scene)
        box = box_tuple(bbox, information["width"], information["height"])
        with Image.open(self.scene_path(scene)) as image:
            crop = image.crop(box).convert("RGB")
        return self._add_example(concept, crop, label, scene, rect_dict(box))

    def add_file(self, concept: str, data: bytes, label: int) -> dict:
        picture = read_picture(data)
        if min(picture.size) < 8:
            raise UserError("نمونه باید حداقل ۸ در ۸ پیکسل باشد.")
        return self._add_example(concept, picture, label)

    def delete_example(self, id: str) -> None:
        path = self.example_path(id)
        with self.connect() as db:
            row = db.execute("SELECT concept FROM examples WHERE id=?", (id,)).fetchone()
            db.execute("DELETE FROM examples WHERE id=?", (id,))
            db.execute("UPDATE concepts SET revision=revision+1 WHERE id=?", (row["concept"],))
        path.unlink(missing_ok=True)

    def example_size(self, concept: str) -> tuple[int, int]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT width,height FROM examples WHERE concept=? AND label=1", (concept,)
            ).fetchall()
        if not rows:
            return 96, 96
        return max(8, round(float(np.median([row[0] for row in rows])))), max(
            8, round(float(np.median([row[1] for row in rows])))
        )

    def train(self, concept: str, job: Job) -> dict:
        current = self.concept(concept)
        with self.connect() as db:
            examples = [
                dict(row)
                for row in db.execute("SELECT * FROM examples WHERE concept=? ORDER BY created", (concept,))
            ]
        positive, negative = sum(row["label"] for row in examples), sum(not row["label"] for row in examples)
        if positive < 2 or negative < 2:
            raise UserError(
                "برای شروع حداقل ۲ نمونهٔ درست و ۲ نمونهٔ غلط مشخص کنید؛ نمونه‌های متنوع‌تر نتیجه را بهتر می‌کنند."
            )
        matrix, labels = [], []
        for index, example in enumerate(examples):
            job.update(index / len(examples) * 85, f"یادگیری از نمونهٔ {index + 1} از {len(examples)}")
            if example["vectors"] and example["feature_version"] == INDEX_VERSION:
                vectors = np.frombuffer(example["vectors"], dtype="<f4").reshape(-1, 512)
            else:
                with Image.open(self.example_path(example["id"])) as original:
                    original = original.convert("RGB")
                    pictures = [original.rotate(angle, expand=True) for angle in (0, 90, 180, 270)]
                    pictures.extend([ImageOps.mirror(original), ImageOps.flip(original)])
                    vectors = np.asarray(self.engine.images(pictures), dtype=np.float32)
                if vectors.shape != (6, 512) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid example embeddings")
                with self.connect() as db:
                    db.execute(
                        "UPDATE examples SET vectors=?,feature_version=? WHERE id=?",
                        (vectors.astype("<f4").tobytes(), INDEX_VERSION, example["id"]),
                    )
            matrix.append(vectors)
            labels.extend([example["label"]] * len(vectors))
        job.update(90, "آموزش تشخیص هدف با نمونه‌های درست و غلط")
        classifier = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=0)
        classifier.fit(np.concatenate(matrix), np.asarray(labels))
        width, height = self.example_size(concept)
        model = {
            "weights": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "feature_version": INDEX_VERSION,
            "window_width": width,
            "window_height": height,
            "positive": positive,
            "negative": negative,
            "trained_at": time.time(),
        }
        job.check()
        with self.connect() as db:
            changed = db.execute(
                "UPDATE concepts SET model=?,trained_revision=? WHERE id=? AND revision=?",
                (json.dumps(model), current["revision"], concept, current["revision"]),
            ).rowcount
            if not changed:
                raise UserError("نمونه‌ها هنگام آموزش تغییر کردند. آموزش را دوباره اجرا کنید.")
        return {
            "concept": concept,
            "message": "آموزش تمام شد. حالا همین هدف را روی تصویر خودت جست‌وجو کن.",
            "positive": positive,
            "negative": negative,
        }

    def plan(
        self, scene: str, width: int, height: int, scope: dict | None = None, thorough: bool = False
    ) -> dict:
        information = self.scene(scene)
        box = (
            box_tuple(scope, information["width"], information["height"])
            if scope
            else (0, 0, information["width"], information["height"])
        )
        if not 8 <= width <= 2048 or not 8 <= height <= 2048:
            raise UserError("اندازهٔ هدف باید بین ۸ تا ۲۰۴۸ پیکسل باشد.")
        windows = []
        seen = set()
        for scale in (0.8, 1.0, 1.25):
            w, h = max(8, round(width * scale)), max(8, round(height * scale))
            if w > box[2] - box[0] or h > box[3] - box[1]:
                continue
            for y in positions(box[1], box[3], h, 0.4 if thorough else 0.65):
                for x in positions(box[0], box[2], w, 0.4 if thorough else 0.65):
                    window = (x, y, x + w, y + h)
                    if window not in seen:
                        seen.add(window)
                        windows.append(window)
                        if len(windows) > MAX_WINDOWS:
                            return {
                                "windows": len(windows),
                                "allowed": False,
                                "limit": MAX_WINDOWS,
                                "scope": rect_dict(box),
                            }
        if not windows:
            raise UserError("محدوده از اندازهٔ هدف کوچک‌تر است. محدوده را بزرگ‌تر یا اندازهٔ هدف را کوچک‌تر کنید.")
        return {
            "windows": len(windows),
            "allowed": True,
            "limit": MAX_WINDOWS,
            "scope": rect_dict(box),
            "boxes": windows,
        }

    def search(
        self,
        scene: str,
        concept: str | None,
        text: str,
        width: int,
        height: int,
        threshold: float,
        scope: dict | None,
        thorough: bool,
        job: Job,
    ) -> dict:
        plan = self.plan(scene, width, height, scope, thorough)
        if not plan["allowed"]:
            raise UserError(
                "تعداد پنجره‌ها بیش از ۵۰۰۰ است. با ابزار «محدوده» یک بخش کوچک‌تر انتخاب کن یا اندازهٔ هدف را بزرگ‌تر کن؛ هیچ بخشی خودکار حذف نمی‌شود."
            )
        model, revision = None, None
        if concept:
            current = self.concept(concept)
            if not current["model"] or current["trained_revision"] != current["revision"]:
                raise UserError("نمونه‌ها جدید شده‌اند. ابتدا آموزش را به‌روزرسانی کن.")
            model = json.loads(current["model"])
            if model["feature_version"] != INDEX_VERSION:
                raise UserError("نسخهٔ ویژگی‌ها تغییر کرده است. آموزش را دوباره اجرا کن.")
            revision = current["revision"]
        elif len(text.strip()) < 3:
            raise UserError("یک هدف آموزش‌دیده انتخاب کن یا توضیحی برای جست‌وجو بنویس.")
        windows = plan["boxes"]
        key = hashlib.sha256(json.dumps([scene, windows, INDEX_VERSION]).encode()).hexdigest()
        cached = self.cache / f"{key}.npy"
        features = None
        if cached.is_file():
            try:
                candidate = np.load(cached, allow_pickle=False)
                if candidate.shape == (len(windows), 512) and np.isfinite(candidate).all():
                    features = candidate
            except (OSError, ValueError):
                pass
        if features is None:
            chunks = []
            with Image.open(self.scene_path(scene)) as original:
                for start in range(0, len(windows), 24):
                    job.update(
                        3 + start / len(windows) * 86,
                        f"بررسی تصویر: {min(start + 24, len(windows))} از {len(windows)} پنجره",
                    )
                    pictures = [original.crop(box).convert("RGB") for box in windows[start : start + 24]]
                    vectors = np.asarray(self.engine.images(pictures), dtype=np.float32)
                    if vectors.shape != (len(pictures), 512) or not np.isfinite(vectors).all():
                        raise ValueError("Invalid search embeddings")
                    chunks.append(vectors)
            features = np.concatenate(chunks)
            temporary = cached.with_suffix(".tmp")
            with temporary.open("wb") as output:
                np.save(output, features, allow_pickle=False)
            temporary.replace(cached)
            caches = sorted(self.cache.glob("*.npy"), key=lambda path: path.stat().st_mtime)
            for old in caches[:-8]:
                old.unlink(missing_ok=True)
        job.update(92, "مقایسه با آموزش شما و حذف کادرهای تکراری")
        if model:
            scores = expit(features @ np.asarray(model["weights"], dtype=np.float32) + model["intercept"])
        else:
            scores = np.clip((features @ self.engine.text(text.strip()) + 1) / 2, 0, 1)
        results = suppress(windows, scores, threshold)
        identifier = uuid.uuid4().hex
        settings = {
            "scene": scene,
            "concept": concept,
            "text": text.strip(),
            "width": width,
            "height": height,
            "threshold": threshold,
            "scope": plan["scope"],
            "windows": len(windows),
            "thorough": thorough,
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                (
                    identifier,
                    scene,
                    concept,
                    revision,
                    json.dumps(settings),
                    json.dumps(results),
                    time.time(),
                ),
            )
        return {
            "run": identifier,
            "scene": scene,
            "concept": concept,
            "revision": revision,
            "settings": settings,
            "detections": results,
            "message": f"جست‌وجو تمام شد؛ {len(results)} کادر پیشنهادی پیدا شد.",
        }

    def run(self, id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (id,)).fetchone()
        if row is None:
            raise UserError("نتیجه پیدا نشد.")
        return {
            **dict(row),
            "settings": json.loads(row["settings"]),
            "detections": json.loads(row["results"]),
        }

    def crop_detection(self, run: str, index: int) -> Image.Image:
        result = self.run(run)
        if not 0 <= index < len(result["detections"]):
            raise UserError("کادر نتیجه پیدا نشد.")
        scene = self.scene(result["scene"])
        box = box_tuple(result["detections"][index]["bbox"], scene["width"], scene["height"])
        with Image.open(self.scene_path(scene["id"])) as image:
            return image.crop(box).convert("RGB")

    def export_training(self, concept: str) -> bytes:
        current = self.concept(concept)
        entries = self.examples(concept)
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = {"format": "atlas-training-v1", "name": current["name"], "examples": []}
            for example in entries:
                filename = f"examples/{example['id']}.png"
                archive.write(self.example_path(example["id"]), filename)
                manifest["examples"].append(
                    {
                        "file": filename,
                        "label": example["label"],
                        "width": example["width"],
                        "height": example["height"],
                    }
                )
            archive.writestr("training.json", json.dumps(manifest, ensure_ascii=False))
        return data.getvalue()

    def import_training(self, data: bytes, job: Job) -> dict:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise UserError("بستهٔ آموزشی ZIP معتبر نیست.") from exc
        with archive:
            infos = archive.infolist()
            if (
                len(infos) > 501
                or sum(item.file_size for item in infos) > 200 * 1024 * 1024
                or len({i.filename for i in infos}) != len(infos)
            ):
                raise UserError("بستهٔ آموزشی بیش از حد بزرگ یا دارای نام تکراری است.")
            try:
                if archive.getinfo("training.json").file_size > 1_000_000:
                    raise ValueError()
                manifest = json.loads(archive.read("training.json"))
                if (
                    manifest["format"] != "atlas-training-v1"
                    or not 1 <= len(manifest["examples"]) <= MAX_EXAMPLES
                ):
                    raise ValueError()
                name = manifest["name"].strip()
                if not 1 <= len(name) <= 100:
                    raise ValueError()
                prepared = []
                total_pixels = 0
                for number, example in enumerate(manifest["examples"]):
                    job.update(number / len(manifest["examples"]) * 40, "بررسی نمونه‌های بستهٔ آموزشی")
                    file = example["file"]
                    path = PurePosixPath(file)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or "\\" in file
                        or ":" in file
                        or example["label"] not in (0, 1)
                    ):
                        raise ValueError()
                    if archive.getinfo(file).file_size > 15 * 1024 * 1024:
                        raise ValueError()
                    picture = read_picture(archive.read(file))
                    total_pixels += picture.width * picture.height
                    if total_pixels > MAX_PIXELS:
                        raise UserError(
                            "مجموع تصاویر بسته بیش از ۶۴ میلیون پیکسل است؛ بسته را به چند بخش تقسیم کنید."
                        )
                    original_size = (
                        example.get("width", picture.width),
                        example.get("height", picture.height),
                    )
                    if any(type(size) is not int or not 8 <= size <= 16000 for size in original_size):
                        raise ValueError()
                    picture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    prepared.append((picture, example["label"], original_size))
            except (KeyError, ValueError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
                raise UserError("ساختار بستهٔ آموزشی معتبر نیست.") from exc
            concept = self.new_concept(name)
            for number, (picture, label, original_size) in enumerate(prepared):
                job.update(45 + number / len(prepared) * 50, "واردکردن نمونه‌های آموزشی")
                entry = self._add_example(concept["id"], picture, label)
                with self.connect() as db:
                    db.execute(
                        "UPDATE examples SET width=?,height=? WHERE id=?", (*original_size, entry["id"])
                    )
        return {
            "concept": concept["id"],
            "message": "نمونه‌ها وارد شدند. برای این هدف یک‌بار دکمهٔ آموزش را بزن.",
        }


def positions(start: int, end: int, size: int, fraction: float) -> list[int]:
    result = list(range(start, end - size + 1, max(4, round(size * fraction))))
    last = end - size
    if not result or result[-1] != last:
        result.append(last)
    return result


def suppress(boxes: list, scores: np.ndarray, threshold: float, limit: int = 200) -> list[dict]:
    order = np.argsort(-scores, kind="stable")
    chosen = []
    for index in order:
        if scores[index] < threshold:
            break
        box = boxes[index]
        if any(intersection_over_union(box, selected[0]) > 0.3 for selected in chosen):
            continue
        chosen.append((box, float(scores[index])))
        if len(chosen) >= limit:
            break
    return [
        {"index": number, "bbox": rect_dict(box), "score": round(score, 5)}
        for number, (box, score) in enumerate(chosen)
    ]


def intersection_over_union(a, b) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
