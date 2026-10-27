"""Real, offline functional checks for precision tools; no satellite accuracy claims."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .engine import Engine
from .geometry import pose_proposals, verify
from .jobs import Job
from .reference_matcher import ReferenceMatcher
from .teaching_validation import make_teaching_scene
from .validation import deny_external_network
from .visual_features import model_picture, shape_picture, unit
from .workshop import Workshop


def validate_precision(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with deny_external_network() as attempts:
        engine = Engine(root)
        engine.configure("cpu", "maximum")
        workshop = Workshop(root, engine)
        similar = workshop.similarity
        picture, examples, truth, distractors = make_teaching_scene()
        source = root / "precision-scene.png"
        picture.save(source)
        scene = workshop.import_scene_file(source, source.name, Job("import"))["scene"]
        group = similar.library.create("precision validation")["id"]
        positive = similar.add_file(examples[0], "target.png")
        similar.library.add(group, positive["id"])
        negative = similar.add_box(scene["id"], dict(x=576, y=32, width=128, height=96))
        similar.library.add(group, negative["id"], 0)
        test_positive = similar.add_box(scene["id"], dict(x=32, y=40, width=128, height=96))
        similar.library.add(group, test_positive["id"], 1, "test")
        test_negative = similar.add_reference(
            Image.fromarray(np.random.default_rng(819).integers(0, 256, (96, 128, 3), dtype=np.uint8)),
            "held-out negative",
        )
        similar.library.add(group, test_negative["id"], 0, "test")
        trained = similar.learning.train(Job("train"), group)
        state = similar.learning.status(group)
        assert state["ready"] and trained["report"]["evaluation_references"] == 2
        assert trained["report"]["metrics"] is not None
        helper = ReferenceMatcher(
            similar,
            dict(references=[], negative_references=[], rotate=True, broad=False, intelligent=False),
            Job("features"),
        )
        pos, neg = [helper.features(r["id"]) for r in (positive, negative)]
        scores = [
            float(
                np.mean(
                    [
                        similar.learning.predict(state["model"], p, a)
                        for p, a in zip(v["pooled"], v["aerial"], strict=True)
                    ]
                )
            )
            for v in (pos, neg)
        ]
        assert scores[0] > scores[1], scores
        with Image.open(similar.folder / f"{positive['id']}.png") as reference:
            ref = reference.convert("RGB")
        draft = similar.segmentation.propose(
            Job("segment"),
            positive["id"],
            points=[dict(x=35, y=30, label=1), dict(x=5, y=5, label=0)],
            box=dict(x=15, y=10, width=96, height=77),
        )
        with Image.open(similar.segmentation.folder / f"{draft['draft']}.png") as mask:
            values = np.asarray(mask)
            assert values[30, 35] > 0 and values[0, 0] == 0 and 0.05 < (values > 0).mean() < 0.9
        cached_embedding = engine._segment_embeddings[1]
        engine.segment(ref, points=[dict(x=35, y=30, label=1)])
        assert engine._segment_embeddings[1] is cached_embedding
        masked = similar.segmentation.apply(group, positive["id"], draft["draft"])
        variant = next(r for r in masked["references"] if r["has_mask"])
        assert (
            variant["source_reference"] == positive["id"]
            and similar.segmentation.mask(variant["id"]) is not None
        )
        similar.library.replace(group, variant["id"], positive["id"])
        assert similar.learning.status(group)["ready"]
        prepared = model_picture(shape_picture(ref))
        feature_key = "precision:" + similar.feature_store.fingerprint(scene["id"])
        expected_features = similar.feature_store.cached(feature_key, prepared)
        restarted = Workshop(root, engine)
        cached_features = restarted.similarity.feature_store.cached(feature_key, prepared)
        assert restarted.similarity.feature_store.hits == 1
        assert np.allclose(expected_features["grid"], cached_features["grid"], atol=0.001)
        similar.feature_store.build(Job("index"), scene["id"], 512)
        index = similar.feature_store.index_plan(scene["id"], 512)
        assert index["windows"] == index["completed"]
        found = similar.feature_store.shortlist(
            scene["id"], pos["pooled"], 512, 2, dict(x=0, y=0, width=100, height=100)
        )
        assert found and all(r["x"] < 100 and r["y"] < 100 for r in found)
        run = similar.search(
            Job("fast"),
            scene=scene["id"],
            group=group,
            width=128,
            height=96,
            threshold=0.55,
            rotate=False,
            mode="fast",
            fast_regions=10,
        )
        assert run["settings"]["mode"] == "fast" and run["settings"]["selected_tiles"]
        assert run["detections"], "Actual models produced no controlled target proposals"
        feedback = similar.learning.feedback(Job("feedback"), run["run"], run["detections"][0]["index"], 1)
        assert feedback["teaching"] == group and similar.results(run["run"])["detections"][0]["feedback"] == 1

        # A continuous angle and projective warp, independent of the 45-degree templates.
        textured = cv2.GaussianBlur(
            np.random.default_rng(91).integers(0, 256, (160, 200), dtype=np.uint8), (3, 3), 0
        )
        geometry_reference = Image.fromarray(textured).convert("RGB")
        matrix = cv2.getRotationMatrix2D((100, 80), 17, 1.12)
        matrix[:, 2] += [85, 70]
        homography = np.vstack([matrix, [0.0003, -0.0002, 1.0]])
        candidate = Image.fromarray(
            cv2.warpPerspective(np.asarray(geometry_reference), homography, (400, 320))
        )
        proof = verify(geometry_reference, candidate)
        assert proof["status"] == "verified" and proof["inliers"] >= 10
        assert abs(((proof["rotation"] + 17 + 180) % 360) - 180) < 5
        assert list(pose_proposals(geometry_reference, None, candidate, lambda: None))

        import onnxruntime as ort

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = engine.compute.cpu_threads()
        session = ort.InferenceSession(
            str(engine.compute.graph), sess_options=options, providers=["CPUExecutionProvider"]
        )
        onnx_errors = []
        for size in [(224, 224), (224, 168), (448, 448)]:
            p = ref.resize(size)
            pixels = np.asarray(p, np.float32).transpose(2, 0, 1)[None] / 255
            pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )[None, :, None, None]
            tokens = unit(session.run(["tokens"], {"pixels": pixels})[0][0])
            actual = engine.details([p])[0]
            expected = np.concatenate([actual["cls"][None], actual["grid"].reshape(-1, 384)])
            assert np.allclose(tokens, expected, atol=0.001, rtol=0.001)
            onnx_errors.append(float(np.max(np.abs(tokens - expected))))
        engine.configure("auto", "maximum")
        engine.details([prepared])
        hardware = engine.status()["compute"]
        assert not attempts
        report = dict(
            status="passed",
            offline=True,
            external_python_connections=len(attempts),
            aerial_model="RemoteCLIP ViT-B-32 vision",
            segmentation="SAM2.1 Hiera Tiny image model",
            target_training_scores=scores,
            evaluation=trained["report"],
            segmentation_prompt_cache_reused=True,
            immutable_original=True,
            feature_cache_restart_reused=True,
            index=index,
            approximate_search_results=run["total_results"],
            feedback_persisted=True,
            perspective_inliers=proof["inliers"],
            continuous_rotation=proof["rotation"],
            onnx_cpu_parity_max_error=max(onnx_errors),
            hardware=hardware,
            gpu_hardware_exercised=hardware["device"] != "CPU",
            limitation="Controlled functional fixtures; no satellite benchmark or physical one-terabyte dataset speed measurement",
        )
    (root / "precision-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
