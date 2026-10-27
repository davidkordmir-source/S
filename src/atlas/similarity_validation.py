"""Reproducible screenshot matching fixture; not a satellite accuracy benchmark."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_similarity_scene():
    random = np.random.default_rng(261027)
    sample = Image.fromarray(random.integers(25, 160, (48, 64, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(sample)
    draw.rectangle((6, 5, 25, 35), fill="#e8d5b0")
    draw.rectangle((36, 17, 56, 41), fill="#b93926")
    draw.line((0, 40, 63, 8), fill="#eff1ed", width=4)
    scene = Image.fromarray(random.integers(55, 125, (560, 1800, 3), dtype=np.uint8))
    truth = []
    for x, y, scale, angle in [
        (100, 100, 1, 0),
        (752, 170, 1, 90),
        (1500, 390, 1.25, 180),
        (1200, 90, 1, 45),
    ]:
        picture = sample.resize((round(64 * scale), round(48 * scale)), Image.Resampling.LANCZOS).rotate(
            angle, expand=True
        )
        # JPEG screenshot and slight exposure differences.
        picture = Image.fromarray(np.clip(np.asarray(picture).astype(np.int16) + 7, 0, 255).astype(np.uint8))
        scene.paste(picture, (x, y))
        truth.append((x, y, x + picture.width, y + picture.height))
    stream = io.BytesIO()
    sample.save(stream, format="JPEG", quality=87)
    return scene, stream.getvalue(), truth


def validate_similarity(root: Path):
    from .engine import Engine
    from .jobs import Job
    from .validation import deny_external_network
    from .workshop import Workshop, box_tuple, intersection_over_union

    root.mkdir(parents=True, exist_ok=True)
    with deny_external_network() as attempts:
        engine = Engine(root)
        workshop = Workshop(root, engine)
        scene, data, truth = make_similarity_scene()
        source = root / "similarity-scene.png"
        scene.save(source)
        info = workshop.import_scene_file(source, source.name, Job("import"))["scene"]
        reference = workshop.similarity.add_file(data, "reference-screenshot.jpg")
        options = dict(
            scene=info["id"],
            reference=reference["id"],
            width=64,
            height=48,
            threshold=0.78,
            rotate=True,
            intelligent=True,
        )
        result = workshop.similarity.search(Job("similar"), **options)
        boxes = [box_tuple(d["bbox"], scene.width, scene.height) for d in workshop.iter_detections(result)]
        matches = [any(intersection_over_union(b, target) > 0.5 for b in boxes) for target in truth]
        assert all(matches), (matches, boxes)
        assert len(boxes) <= 10, boxes
        repeated = workshop.similarity.search(Job("similar"), **options)
        assert repeated["total_results"] == result["total_results"]
        assert not attempts
        report = dict(
            status="passed",
            offline=True,
            actual_bundled_embeddings=True,
            reference_examples=1,
            negative_examples_required=False,
            known_patterns=len(truth),
            matched_patterns=sum(matches),
            detections=len(boxes),
            rotation_45_degree_steps=True,
            scale_variation=True,
            tile_boundary_match=True,
            cached_rescan=True,
            external_connections=len(attempts),
            note="Controlled matching fixture, not measured satellite detection accuracy",
        )
    (root / "similarity-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
