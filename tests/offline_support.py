"""Deterministic doubles for API/UI wiring; never used by the shipped app."""

import io

import numpy as np
from PIL import Image, ImageDraw


def scene_bytes():
    picture = Image.new("RGB", (256, 192), "#647b60")
    draw = ImageDraw.Draw(picture)
    for x, y, color in [
        (16, 16, "#df3024"),
        (144, 16, "#ca3529"),
        (16, 112, "#255ed1"),
        (144, 112, "#1c55be"),
    ]:
        draw.rectangle((x, y, x + 63, y + 63), fill=color)
    output = io.BytesIO()
    picture.save(output, "PNG")
    return output.getvalue()


class OfflineTestEngine:
    def __init__(self):
        self.calls = 0
        self.hardware = dict(
            preference="auto",
            performance="maximum",
            cpu_threads=1,
            device="CPU",
            gpu_available=False,
            adapters=[],
            fallback_reason="",
        )

    def configure(self, device, performance=None):
        self.hardware.update(preference=device, performance=performance or "maximum")
        return self.hardware

    def status(self):
        return dict(
            image_available=True,
            text_available=True,
            image_loaded=True,
            text_loaded=True,
            offline=True,
            detail_available=True,
            aerial_available=True,
            segmentation_available=True,
            compute=self.hardware,
            device="CPU",
        )

    def images(self, pictures):
        self.calls += len(pictures)
        vectors = np.zeros((len(pictures), 512), dtype=np.float32)
        for index, picture in enumerate(pictures):
            vectors[index, :3] = np.asarray(picture.convert("RGB")).mean(axis=(0, 1)) / 255
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def text(self, query):
        vector = np.zeros(512, dtype=np.float32)
        vector[0 if "قرمز" in query or "red" in query else 2] = 1
        return vector

    def details(self, pictures):
        # Wiring double only. Real DINO retrieval is checked with pinned weights
        # by the packaged --similarity-smoke-test, never by this stand-in.
        return [
            dict(
                cls=np.zeros(384, np.float32), grid=np.zeros((p.height // 14, p.width // 14, 384), np.float32)
            )
            for p in pictures
        ]

    def aerial(self, pictures):
        vectors = np.zeros((len(pictures), 512), np.float32)
        vectors[:, 0] = 1
        return vectors

    def segment(self, picture, points=None, box=None):
        mask = Image.new("L", picture.size)
        draw = ImageDraw.Draw(mask)
        box = box or dict(
            x=picture.width // 4, y=picture.height // 4, width=picture.width // 2, height=picture.height // 2
        )
        draw.rectangle((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]), fill=255)
        return mask, 0.8
