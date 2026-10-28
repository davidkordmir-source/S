"""Native checks on actual local VLM and OWLv2 weights, not test doubles."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import Engine
from .jobs import Job
from .object_detector import ObjectDetector
from .validation import deny_external_network
from .workshop import intersection_over_union


def validate_neural(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    fixture = Path(os.environ["ATLAS_NEURAL_FIXTURE"])
    with Image.open(fixture) as image:
        picture = image.convert("RGB")
    # COCO 000000039769: two different cats, not copy/paste instances.
    width, height = picture.size
    truth = [(0, 40, 335, 475), (335, 0, 640, 430)]
    if (width, height) != (640, 480):
        raise AssertionError("Unexpected official COCO fixture dimensions")
    started = time.monotonic()
    engine = Engine(root)
    try:
        with deny_external_network() as attempts:
            assert engine.objects.status()["available"] and engine.assistant.status()["available"]
            reply = engine.assistant.understand(
                Job("assistant"), "همهٔ گربه‌ها را پیدا کن، حتی اگر رنگ و حالت بدنشان متفاوت باشد.", [picture]
            )
            assert reply["intent"] == "search" and reply["detector_phrases"], reply
            assert any("\u0600" <= c <= "\u06ff" for c in reply["answer_fa"]), reply
            cached = engine.assistant.understand(
                Job("assistant"), "همهٔ گربه‌ها را پیدا کن، حتی اگر رنگ و حالت بدنشان متفاوت باشد.", [picture]
            )
            assert cached["cached"] and engine.assistant.calls == 1
            engine.assistant.close()
            query = picture.crop((5, 45, 330, 475))
            vectors = engine.objects.image_query(query, 768, False)
            scene = engine.objects.scene(picture, 768, False)
            image_result = engine.objects.classify(scene, images=vectors)
            texts = engine.objects.text_queries(["a cat"])
            text_result = engine.objects.classify(scene, texts=texts)
            boxes = ObjectDetector.boxes(scene, width, height)
            best = []
            for target in truth:
                indexes = [
                    i for i, box in enumerate(boxes) if intersection_over_union(tuple(box), target) > 0.3
                ]
                assert indexes, target
                best.append(
                    dict(
                        image_score=float(image_result["scores"][indexes].max()),
                        text_score=float(text_result["scores"][indexes].max()),
                    )
                )
            assert all(row["image_score"] >= 0.05 and row["text_score"] >= 0.1 for row in best), best
            assert not np.array_equal(
                np.asarray(query.resize((64, 64))), np.asarray(picture.crop(truth[1]).resize((64, 64)))
            )
            assert not attempts, attempts
            report = dict(
                status="passed",
                fixture="COCO val2017 000000039769, two nonidentical cats",
                satellite_benchmark=False,
                actual_vlm=reply,
                native_detector_objects=best,
                seconds=time.monotonic() - started,
                external_network_attempts=attempts,
                detector_revision=engine.objects.revision(),
                assistant_revision=engine.assistant.revision(),
            )
        (root / "neural-smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logging.info("Real multimodal inference and nonidentical instance check passed: %s", report)
        return report
    finally:
        engine.close()
