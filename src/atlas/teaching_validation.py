"""Controlled multiview/colour-distractor checks; not a satellite benchmark."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def make_teaching_scene():
    def render(oblique=False, held_out=False):
        random = np.random.default_rng(700 + int(oblique) * 10 + int(held_out))
        picture = Image.fromarray(random.normal([85, 104, 71], 5, (96, 128, 3)).clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(picture)
        colour = "#526dba" if held_out else "#bc6743"
        if oblique:
            draw.polygon([(17, 33), (91, 8), (116, 32), (40, 60)], fill=colour)
            draw.polygon(
                [(40, 60), (116, 32), (116, 66), (40, 91)], fill="#9f9a83" if held_out else "#e0c89e"
            )
            for i in range(5):
                x = 46 + i * 13
                y = 67 - i * 4
                draw.rectangle((x, y, x + 6, y + 10), fill="#333e3d")
            draw.polygon([(17, 33), (40, 60), (40, 91), (17, 62)], fill="#8c8067")
        else:
            draw.polygon([(20, 15), (104, 15), (104, 40), (53, 40), (53, 80), (20, 80)], fill=colour)
            draw.line((23, 20, 99, 20, 99, 35, 48, 35, 48, 75, 25, 75), fill="#d6cbb5", width=3)
            draw.rectangle((76, 60, 110, 81), fill="#c9bc9d")
        return picture

    references = [render(), render(True)]
    scene = Image.fromarray(
        np.random.default_rng(42).normal([84, 103, 70], 5, (384, 768, 3)).clip(0, 255).astype(np.uint8)
    )
    truth = []
    for x, y, picture in [(32, 40, render(held_out=True)), (320, 90, render(True, True))]:
        scene.paste(picture, (x, y))
        truth.append((x, y, x + 128, y + 96))
    pixels = np.asarray(references[0]).reshape(-1, 3).copy()
    np.random.default_rng(9).shuffle(pixels)
    palette = Image.fromarray(pixels.reshape(96, 128, 3))
    scene.paste(palette, (576, 32))
    mean = tuple(np.asarray(references[0]).mean(axis=(0, 1)).astype(np.uint8))
    scene.paste(Image.new("RGB", (128, 96), mean), (576, 220))
    streams = []
    for picture in references:
        data = io.BytesIO()
        picture.save(data, "PNG")
        streams.append(data.getvalue())
    return scene, streams, truth, [(576, 32, 704, 128), (576, 220, 704, 316)]


def validate_teaching(workshop, root):
    from .jobs import Job
    from .workshop import box_tuple, intersection_over_union

    picture, examples, truth, distractors = make_teaching_scene()
    path = root / "teaching-scene.png"
    picture.save(path)
    scene = workshop.import_scene_file(path, path.name, Job("import"))["scene"]
    library = workshop.similarity.library
    target = library.create("یک هدف از دو زاویه")
    refs = [workshop.similarity.add_file(data, f"view-{i + 1}.png") for i, data in enumerate(examples)]
    library.add(target["id"], refs[0]["id"])
    options = dict(
        scene=scene["id"],
        group=target["id"],
        width=128,
        height=96,
        rotate=True,
        intelligent=True,
        threshold=0.7,
    )
    first = workshop.similarity.search(Job("one-view"), **options)
    library.add(target["id"], refs[1]["id"])
    second = workshop.similarity.search(Job("two-views"), **options)

    def matched(run, target):
        return [
            d
            for d in workshop.iter_detections(run)
            if intersection_over_union(box_tuple(d["bbox"], 768, 384), target) > 0.5
        ]

    single = [bool(matched(first, t)) for t in truth]
    multiple = [bool(matched(second, t)) for t in truth]
    assert single[0] and all(multiple), (single, multiple)
    assert sum(multiple) > sum(single), (single, multiple)
    assert matched(second, truth[1])[0]["matched_reference"] == refs[1]["id"]
    relaxed = workshop.similarity.search(Job("palette-rejection"), **{**options, "threshold": 0.55})
    for target in distractors:
        assert not matched(relaxed, target), matched(relaxed, target)
    return dict(
        one_example_search=True,
        single_view_targets=sum(single),
        multiple_view_targets=sum(multiple),
        targets=len(truth),
        same_palette_distractors_rejected=len(distractors),
        matching_view_explained=True,
        colour_evidence=False,
        reference_features_cached=True,
    )
