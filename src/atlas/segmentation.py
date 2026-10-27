"""Prompted target isolation with reviewable mask drafts and immutable reference variants."""

from __future__ import annotations

import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .errors import UserError


class Segmentation:
    def __init__(self, similarity):
        self.similarity = similarity
        self.folder = similarity.folder / "masks"
        self.folder.mkdir(exist_ok=True)
        with similarity.workshop.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS mask_drafts(id TEXT PRIMARY KEY,reference TEXT,created REAL,score REAL)"
            )
            columns = {r[1] for r in db.execute("PRAGMA table_info(image_references)")}
            if "source_reference" not in columns:
                db.execute("ALTER TABLE image_references ADD COLUMN source_reference TEXT")

    def source(self, id):
        reference = self.similarity.reference(id)
        return (
            self.similarity.reference(reference["source_reference"])
            if reference.get("source_reference")
            else reference
        )

    def propose(self, job, id, points=None, box=None, polygon=None):
        ref = self.source(id)
        with Image.open(self.similarity.folder / f"{ref['id']}.png") as original:
            picture = original.convert("RGB")
        if points and any(not (0 <= p["x"] < picture.width and 0 <= p["y"] < picture.height) for p in points):
            raise UserError("نقطهٔ انتخابی بیرون عکس است.")
        if polygon:
            if len(polygon) < 3 or any(
                not (0 <= p["x"] <= picture.width and 0 <= p["y"] <= picture.height) for p in polygon
            ):
                raise UserError("چندضلعی انتخابی معتبر نیست.")
            mask = Image.new("L", picture.size)
            ImageDraw.Draw(mask).polygon([(p["x"], p["y"]) for p in polygon], fill=255)
            quality = None
        else:
            if not points and not box:
                raise UserError("داخل هدف کلیک کن یا دور آن کادر بکش.")
            if box and (box["x"] + box["width"] > picture.width or box["y"] + box["height"] > picture.height):
                raise UserError("کادر انتخابی بیرون عکس است.")
            job.update(5, "جداکردن هدف از پس‌زمینه با مدل آفلاین؛ نتیجه پیش از اعمال نمایش داده می‌شود")
            mask, quality = self.similarity.workshop.engine.segment(picture, points, box)
        job.check()
        bounds = mask.getbbox()
        if not bounds or min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 8:
            raise UserError("هدف واضحی جدا نشد؛ کادر یا نقطه‌ها را اصلاح کن.")
        draft = uuid.uuid4().hex
        mask.save(self.folder / f"{draft}.png")
        overlay = Image.new("RGBA", picture.size, (35, 190, 105, 0))
        overlay.putalpha(mask.point(lambda value: round(value * 0.5)))
        overlay.save(self.folder / f"{draft}-overlay.png")
        with self.similarity.workshop.connect() as db:
            db.execute("INSERT INTO mask_drafts VALUES(?,?,?,?)", (draft, ref["id"], time.time(), quality))
        return dict(
            draft=draft,
            reference=ref["id"],
            mask_url=f"/api/workspace/masks/{draft}/overlay",
            width=picture.width,
            height=picture.height,
            quality=quality,
            message="ماسک آماده است؛ پیش‌نمایش را بررسی و در صورت مناسب‌بودن اعمال کن.",
        )

    def draft(self, id):
        with self.similarity.workshop.connect() as db:
            row = db.execute("SELECT * FROM mask_drafts WHERE id=?", (id,)).fetchone()
        if not row:
            raise UserError("پیش‌نمایش ماسک پیدا نشد.")
        return dict(row)

    def apply(self, group, reference, draft):
        target = self.similarity.library.group(group)
        if not any(r["id"] == reference for r in target["references"]):
            raise UserError("عکس در هدف انتخابی نیست.")
        meta = self.draft(draft)
        source = self.source(reference)
        if meta["reference"] != source["id"]:
            raise UserError("این ماسک متعلق به عکس دیگری است.")
        with (
            Image.open(self.similarity.folder / f"{source['id']}.png") as original,
            Image.open(self.folder / f"{draft}.png") as saved,
        ):
            picture = original.convert("RGB")
            mask = saved.convert("L")
            bounds = mask.getbbox()
            padding = max(2, round(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05))
            bounds = (
                max(0, bounds[0] - padding),
                max(0, bounds[1] - padding),
                min(picture.width, bounds[2] + padding),
                min(picture.height, bounds[3] + padding),
            )
            cropped = picture.crop(bounds)
            foreground = mask.crop(bounds)
            pixels = np.asarray(cropped.convert("L"))
            selected = pixels[np.asarray(foreground) > 0]
            background = int(np.clip(float(selected.mean()) + (-64 if selected.mean() > 127 else 64), 0, 255))
            isolated = Image.composite(cropped, Image.new("RGB", cropped.size, (background,) * 3), foreground)
            size = (
                round(cropped.width * source["width"] / picture.width),
                round(cropped.height * source["height"] / picture.height),
            )
            ref = self.similarity.add_reference(
                isolated, "هدف جداشده · " + source["name"], original_size=size
            )
            foreground.save(self.folder / f"{ref['id']}.png")
        with self.similarity.workshop.connect() as db:
            db.execute("UPDATE image_references SET source_reference=? WHERE id=?", (source["id"], ref["id"]))
        self.similarity.library.replace(group, reference, ref["id"])
        return self.similarity.library.group(group)

    def mask(self, id, size=None):
        path = self.folder / f"{id}.png"
        if not path.is_file():
            return None
        with Image.open(path) as source:
            mask = source.convert("L")
            if size:
                mask = mask.resize(size, Image.Resampling.NEAREST)
            return np.asarray(mask).copy()
