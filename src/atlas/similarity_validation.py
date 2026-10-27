"""Reproducible screenshot matching fixture; not a satellite accuracy benchmark."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_variant_scene():
    """Independently rendered roof shapes, not rotated copies of the query."""

    def roof(variant=0):
        random = np.random.default_rng(450 + variant)
        pixels = random.normal([83, 104, 68], 8, (96, 128, 3)).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        colors = [(186, 69, 43), (55, 84, 151), (152, 148, 120)]
        color = colors[variant % 3]
        shift = variant * 3
        draw.polygon(
            [(21, 21 + shift), (78, 17 + shift), (82, 49 + shift), (57, 51 + shift), (59, 77), (25, 79)],
            fill=color,
        )
        draw.line(
            (24, 25 + shift, 78, 21 + shift, 79, 47 + shift, 54, 48 + shift, 55, 74, 29, 76),
            fill=tuple(min(255, c + 38) for c in color),
            width=3,
        )
        draw.rectangle((94, 24 + shift, 116, 56 + shift), fill=(180 - variant * 20, 177, 155))
        draw.line((0, 89, 127, 81), fill="#bcb69c", width=7)
        if variant:
            draw.ellipse((0, 0, 18 + shift, 25), fill="#426443")
            draw.rectangle((43, 32 + shift, 55, 43 + shift), fill="#d3d2c6")
        return image

    sample = roof()
    scene = Image.fromarray(
        np.random.default_rng(50).normal([82, 103, 69], 12, (384, 768, 3)).clip(0, 255).astype(np.uint8)
    )
    truth = []
    for variant, (x, y) in enumerate([(32, 32), (280, 40), (520, 230)]):
        scene.paste(roof(variant), (x, y))
        truth.append((x, y, x + 128, y + 96))
    # Different geometry and texture as a distractor.
    draw = ImageDraw.Draw(scene)
    draw.ellipse((120, 230, 210, 320), fill="#b3b7a0")
    draw.line((0, 185, 767, 185), fill="#c5c3b3", width=6)
    output = io.BytesIO()
    sample.save(output, "JPEG", quality=85)
    return scene, output.getvalue(), truth


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
        # A second scene is independently rendered with changed roof colours,
        # textures and geometry. This guards against only testing query copies.
        variant_scene, variant_data, variant_truth = make_variant_scene()
        variant_path = root / "similarity-variants.png"
        variant_scene.save(variant_path)
        variant_info = workshop.import_scene_file(variant_path, variant_path.name, Job("import"))["scene"]
        variant_ref = workshop.similarity.add_file(variant_data, "variant-query.jpg")
        variant_result = workshop.similarity.search(
            Job("variants"),
            scene=variant_info["id"],
            reference=variant_ref["id"],
            width=128,
            height=96,
            threshold=0.55,
            intelligent=True,
            rotate=True,
        )
        ranked = list(workshop.iter_detections(variant_result))
        top_boxes = [box_tuple(d["bbox"], 768, 384) for d in ranked[:3]]
        assert all(
            any(intersection_over_union(b, target) > 0.5 for b in top_boxes) for target in variant_truth
        ), ranked[:3]
        assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]
        assert ranked[0]["level"] == "high" and all(d["level"] == "medium" for d in ranked[1:3])
        from .visual_features import dense_proposals, descriptor, model_picture

        with Image.open(io.BytesIO(variant_data)) as query:
            features = [
                descriptor(f)
                for f in engine.details(
                    [model_picture(query.rotate(a, expand=True)) for a in range(0, 360, 45)]
                )
            ]
        proposals = [
            b
            for b, _ in dense_proposals(
                variant_scene, engine, features, 128, 96, (0.8, 1, 1.25), True, lambda: None
            )
        ]
        assert all(
            any(intersection_over_union(b, target) > 0.5 for b in proposals) for target in variant_truth
        )
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
            dense_feature_model="facebook/dinov2-small",
            independently_rendered_targets=len(variant_truth),
            independently_rendered_targets_in_top3=len(variant_truth),
            feature_proposals_without_pixel_gate=True,
            ranked_variant_scores=[round(d["score"] * 100, 1) for d in ranked[:3]],
            level_counts=variant_result["level_counts"],
            external_connections=len(attempts),
            note="Controlled matching fixture, not measured satellite detection accuracy",
        )
    (root / "similarity-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
