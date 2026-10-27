import hashlib
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine
from PIL import Image

from atlas.compute import Compute
from atlas.errors import Cancelled, UserError
from atlas.geometry import pose_proposals, verify
from atlas.jobs import Job
from atlas.offline_server import OfflineServices, create_app
from atlas.reference_matcher import ReferenceMatcher
from atlas.teaching_validation import make_teaching_scene
from atlas.workshop import Workshop


@pytest.fixture
def precision(tmp_path):
    workshop = Workshop(tmp_path, OfflineTestEngine())
    scene_picture, examples, _, _ = make_teaching_scene()
    path = tmp_path / "precision.png"
    scene_picture.save(path)
    scene = workshop.import_scene_file(path, path.name, Job("import"))["scene"]
    target = workshop.similarity.library.create("target")["id"]
    refs = [workshop.similarity.add_file(data, f"reference-{i}.png") for i, data in enumerate(examples)]
    for i, ref in enumerate(refs):
        workshop.similarity.library.add(target, ref["id"], 1 - i)
    return workshop, scene, target, refs


def test_prompt_mask_review_is_immutable_and_reset_merges_members(precision):
    workshop, _, group, refs = precision
    similar = workshop.similarity
    source = (similar.folder / f"{refs[0]['id']}.png").read_bytes()
    draft = similar.segmentation.propose(
        Job("mask"), refs[0]["id"], box=dict(x=16, y=10, width=90, height=70)
    )
    assert not similar.reference(refs[0]["id"])["has_mask"]
    updated = similar.segmentation.apply(group, refs[0]["id"], draft["draft"])
    masked = next(r for r in updated["references"] if r["label"] == 1)
    assert masked["source_reference"] == refs[0]["id"] and masked["has_mask"]
    assert similar.segmentation.mask(masked["id"]).shape == (masked["height"], masked["width"])
    assert (similar.folder / f"{refs[0]['id']}.png").read_bytes() == source
    similar.library.add(group, refs[0]["id"])
    similar.library.replace(group, masked["id"], refs[0]["id"])
    assert len(similar.library.group(group)["references"]) == 2
    with pytest.raises(UserError):
        similar.segmentation.apply(group, refs[1]["id"], draft["draft"])


def test_polygon_mask_and_invalid_prompt_bounds(precision):
    similar = precision[0].similarity
    ref = precision[3][0]["id"]
    points = [dict(x=x, y=y, label=1) for x, y in [(10, 10), (90, 15), (50, 70)]]
    draft = similar.segmentation.propose(Job("mask"), ref, polygon=points)
    assert draft["quality"] is None
    with pytest.raises(UserError):
        similar.segmentation.propose(Job("mask"), ref, points=[dict(x=10000, y=1, label=1)])


def test_index_resume_scope_retrieval_and_cache_repair(precision, monkeypatch):
    workshop, scene, _, _ = precision
    store = workshop.similarity.feature_store

    class Stop(Job):
        def update(self, progress, message):
            if progress > 0:
                self.cancel_event.set()
            super().update(progress, message)

    with pytest.raises(Cancelled):
        store.build(Stop("index"), scene["id"], 256)
    assert store.index_plan(scene["id"], 256)["completed"] == 1
    store.build(Job("index"), scene["id"], 256)
    plan = store.index_plan(scene["id"], 256)
    assert plan["completed"] == plan["windows"]
    vector = np.zeros(384, np.float32)
    vector[0] = 1
    scope = dict(x=0, y=0, width=100, height=100)
    assert all(r["x"] < 100 and r["y"] < 100 for r in store.shortlist(scene["id"], [vector], 256, 2, scope))
    assert store.shortlist(scene["id"], [vector], 256, 2, scope)
    picture = Image.new("RGB", (224, 168))
    store.cached("test", picture)
    store.cached("test", picture)
    assert store.hits == 1 and store.misses == 1
    path = store.folder / (hashlib.sha256(b"test").hexdigest() + ".npz")
    with path.open("wb") as stream:
        np.savez(stream, grid=np.zeros((1, 1, 384)), cls=np.zeros(384))
    assert store.cached("test", picture)["grid"].shape == (12, 16, 384)
    assert store.misses == 2
    monkeypatch.setattr(workshop.engine, "details", lambda _: pytest.fail("Index should be reused"))
    store.build(Job("index"), scene["id"], 256)


def test_index_budget_source_change_and_fast_search_metadata(precision):
    workshop, scene, group, refs = precision
    similar = workshop.similarity
    options = dict(scene=scene["id"], group=group, width=128, height=96, rotate=False, adapt=False)
    with pytest.raises(UserError):
        similar.plan(**options, mode="fast")
    with pytest.raises(UserError):
        similar.feature_store.build(Job("index"), scene["id"], 512, budget_gb=1e-10)
    similar.feature_store.build(Job("index"), scene["id"], 512)
    result = similar.search(Job("fast"), **options, mode="fast", fast_regions=10)
    assert result["settings"]["mode"] == "fast"
    assert result["settings"]["windows"] <= result["settings"]["full_windows"]
    assert result["settings"]["selected_tiles"]
    assert similar.plan(**options, mode="fast", fast_regions=10)["completed"] == result["settings"]["windows"]
    assert similar.plan(**options)["allowed"]  # complete search is independent of the coarse index
    path = workshop.scene_path(scene["id"])
    path.touch()
    with pytest.raises(UserError, match="تغییر"):
        similar.feature_store.index_plan(scene["id"], 512)  # stale source features are never used


def test_target_training_uses_negative_data_and_separate_evaluation(precision, monkeypatch):
    workshop, _, target, refs = precision
    similar = workshop.similarity
    extra = similar.add_reference(
        Image.fromarray(np.random.default_rng(41).integers(0, 256, (48, 64, 3), dtype=np.uint8)), "test"
    )
    similar.library.add(target, extra["id"], 1, "test")

    def features(self, ref):
        label = next(r["label"] for r in similar.library.group(target)["references"] if r["id"] == ref)
        pooled = np.zeros((8, 384), np.float32)
        aerial = np.zeros((8, 512), np.float32)
        pooled[:, label] = 1
        aerial[:, label] = 1
        return dict(pooled=pooled, aerial=aerial)

    monkeypatch.setattr(ReferenceMatcher, "features", features)
    similar.learning.train(Job("train"), target)
    state = similar.learning.status(target)
    assert state["ready"] and state["report"]["metrics"] is None  # positive-only test does not invent F1
    a = features(None, refs[0]["id"])
    b = features(None, refs[1]["id"])
    assert similar.learning.predict(state["model"], a["pooled"][0], a["aerial"][0]) > 0.8
    assert similar.learning.predict(state["model"], b["pooled"][0], b["aerial"][0]) < 0.2
    second_test = similar.add_reference(
        Image.fromarray(np.random.default_rng(45).integers(0, 256, (48, 64, 3), dtype=np.uint8)),
        "negative-test",
    )
    similar.library.add(target, second_test["id"], 0, "test")
    report = similar.learning.train(Job("train"), target)["report"]
    assert report["evaluation_kind"] == "independent" and report["evaluation_references"] == 2
    assert (
        report["metrics"]["f1"] == 1
    )  # separable synthetic vectors test the evaluator, not satellite accuracy
    similar.library.change(target, extra["id"], enabled=False)
    assert not similar.learning.status(target)["ready"]
    restarted = Workshop(workshop.folder.parent, workshop.engine).similarity
    assert restarted.learning.status(target)["stale"]


def test_evaluation_excludes_known_overlapping_scene_crops(precision):
    workshop, scene, target, _ = precision
    similar = workshop.similarity
    for partition, x in [("train", 30), ("test", 35)]:
        ref = similar.add_box(scene["id"], dict(x=x, y=40, width=128, height=96))
        similar.library.add(target, ref["id"], 1, partition)
    report = similar.learning.train(Job("train"), target)["report"]
    assert report["excluded_overlapping_test_references"] == 1
    assert report["metrics"] is None and report["evaluation_references"] == 0
    assert "هم‌پوشانی" in report["evaluation_note"]


def test_result_feedback_changes_same_target_and_persists(precision):
    workshop, scene, target, refs = precision
    similar = workshop.similarity
    similar.library.change(target, refs[1]["id"], enabled=False)
    run = similar.search(
        Job("search"), scene=scene["id"], group=target, width=128, height=96, intelligent=False, threshold=0.7
    )
    assert run["detections"]
    feedback = similar.learning.feedback(Job("teach"), run["run"], 0, 0)
    assert feedback["teaching"] == target
    assert similar.results(run["run"])["detections"][0]["feedback"] == 0
    count = len(similar.library.group(target)["references"])
    similar.learning.feedback(Job("teach"), run["run"], 0, 1)
    assert len(similar.library.group(target)["references"]) == count
    assert similar.results(run["run"])["detections"][0]["feedback"] == 1


def test_continuous_pose_and_perspective_verified_with_inliers():
    rng = np.random.default_rng(91)
    image = rng.integers(0, 256, (160, 200), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    reference = Image.fromarray(image).convert("RGB")
    matrix = cv2.getRotationMatrix2D((100, 80), 17, 1.12)
    matrix[:, 2] += [85, 70]
    homography = np.vstack([matrix, [0.0003, -0.0002, 1.0]])
    candidate = Image.fromarray(
        cv2.warpPerspective(np.asarray(reference), homography, (400, 320), borderValue=(90, 90, 90))
    )
    proof = verify(reference, candidate)
    assert proof["status"] == "verified" and proof["inliers"] >= 10
    assert abs(((proof["rotation"] + 17 + 180) % 360) - 180) < 5
    assert list(pose_proposals(reference, None, candidate, lambda: None))
    assert verify(reference, Image.new("RGB", candidate.size))["status"] == "insufficient"


def test_compute_preferences_and_directml_failure_are_observable(tmp_path, monkeypatch):
    compute = Compute(tmp_path, tmp_path / "model.onnx")
    monkeypatch.setattr(compute, "capabilities", lambda: (False, False))
    with pytest.raises(UserError):
        compute.configure("gpu")
    compute.configure("cpu", "balanced")
    assert Compute(tmp_path, compute.graph).preference == "cpu"
    compute.configure("auto", "maximum")
    monkeypatch.setattr(compute, "capabilities", lambda: (False, True))
    compute.adapters = [dict(id=0)]
    fake = SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
    )
    fake.InferenceSession = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("driver failed"))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    assert compute.details(np.zeros((1, 3, 224, 224), np.float32)) is None
    assert compute.status()["fallback_reason"] and compute.effective == "CPU"
    assert compute.adapters == [dict(id=0)]
    assert compute.details(np.zeros((1, 3, 224, 224), np.float32)) is None
    compute.configure("gpu")
    with pytest.raises(UserError):
        compute.details(np.zeros((1, 3, 224, 224), np.float32))


def test_upload_query_labels_and_test_partition_are_validated(tmp_path):
    services = OfflineServices(tmp_path, engine=OfflineTestEngine())
    _, examples, _, _ = make_teaching_scene()
    with TestClient(create_app(services=services, token="test")) as client:
        client.get("/?token=test")
        response = client.post("/api/workspace/references?label=0&partition=test", content=examples[0])
        assert response.status_code == 200, response.text
        target = response.json()["group"]
        member = client.get(f"/api/workspace/reference-groups/{target}").json()["references"][0]
        assert member["label"] == 0 and member["partition"] == "test"
        assert client.post("/api/workspace/references?label=3", content=examples[0]).status_code == 422
        assert client.post("/api/compute", json={"device": "invalid"}).status_code == 422
        assert (
            json.loads(client.get(f"/api/workspace/reference-groups/{target}/learning").text)["negative"] == 0
        )


def test_uniform_background_is_usable_as_negative_but_cannot_be_target(precision):
    similar = precision[0].similarity
    reference = similar.add_reference(Image.new("RGB", (64, 48), "#447744"), "empty background")
    settings = dict(
        references=[], negative_references=[reference], rotate=False, broad=False, intelligent=False
    )
    matcher = ReferenceMatcher(similar, settings, Job("background"))
    assert matcher.features(reference["id"])["aerial"].shape == (1, 512)
    target_matcher = ReferenceMatcher(similar, {**settings, "negative_references": []}, Job("target"))
    with pytest.raises(UserError):
        target_matcher.sample(reference["id"])
