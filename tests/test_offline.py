import io
import json
import threading
import zipfile

import numpy as np
import pytest
from conftest import wait_job
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine, scene_bytes
from PIL import Image

from atlas.errors import UserError
from atlas.jobs import Job
from atlas.offline_server import OfflineServices, create_app
from atlas.workshop import Workshop, intersection_over_union, rect_dict, suppress


@pytest.fixture
def workspace(tmp_path):
    return Workshop(tmp_path, OfflineTestEngine())


def teach(workspace):
    scene = workspace.import_scene(scene_bytes(), "user-satellite.png", Job("import"))["scene"]["id"]
    concept = workspace.new_concept("سقف قرمز")["id"]
    for x, y, label in [(16, 16, 1), (144, 16, 1), (16, 112, 0), (144, 112, 0)]:
        workspace.add_box(concept, scene, dict(x=x, y=y, width=64, height=64), label)
    return scene, concept


def test_training_search_feedback_cache_and_persistence(workspace):
    scene, concept = teach(workspace)
    workspace.train(concept, Job("train"))
    assert workspace.concepts()[0]["ready"]
    assert workspace.engine.calls == 24
    workspace.train(concept, Job("train"))
    assert workspace.engine.calls == 24  # Reuses features, still fits the classifier.
    result = workspace.search(scene, concept, "", 64, 64, 0.6, None, False, Job("search"))
    assert result["detections"]
    first = result["detections"][0]
    assert first["bbox"]["y"] < 90  # Red examples are in the top half.
    calls = workspace.engine.calls
    repeated = workspace.search(scene, concept, "", 64, 64, 0.5, None, False, Job("search"))
    assert workspace.engine.calls == calls
    assert len(repeated["detections"]) >= len(result["detections"])
    workspace.add_box(concept, scene, first["bbox"], 0)
    assert not workspace.concepts()[0]["ready"]
    with pytest.raises(UserError, match="آموزش"):
        workspace.search(scene, concept, "", 64, 64, 0.5, None, False, Job("search"))
    workspace.train(concept, Job("train"))
    reopened = Workshop(workspace.root, workspace.engine)
    assert reopened.concepts()[0]["ready"]
    assert len(reopened.examples(concept)) == 5
    assert reopened.run(result["run"])["detections"] == result["detections"]


def test_requires_both_labels_and_valid_boxes(workspace):
    scene = workspace.import_scene(scene_bytes(), "image.png", Job("import"))["scene"]["id"]
    concept = workspace.new_concept("target")["id"]
    with pytest.raises(UserError, match="۲ نمونه"):
        workspace.train(concept, Job("train"))
    for box in [
        dict(x=250, y=0, width=64, height=64),
        dict(x=0, y=0, width=7, height=64),
        dict(x=float("nan"), y=0, width=64, height=64),
    ]:
        with pytest.raises(UserError):
            workspace.add_box(concept, scene, box, 1)
    assert not workspace.examples(concept)


def test_plan_covers_edges_and_respects_roi_and_budget(workspace):
    scene, _ = teach(workspace)
    roi = dict(x=20, y=30, width=190, height=130)
    plan = workspace.plan(scene, 64, 64, roi)
    assert plan["allowed"]
    assert any(box[2] == 210 for box in plan["boxes"])
    assert any(box[3] == 160 for box in plan["boxes"])
    assert all(20 <= box[0] < box[2] <= 210 and 30 <= box[1] < box[3] <= 160 for box in plan["boxes"])
    denied = workspace.plan(scene, 8, 8, thorough=True)
    assert not denied["allowed"] and denied["windows"] == 5001 and "boxes" not in denied
    assert workspace.plan(scene, 8, 8, dict(x=20, y=20, width=40, height=40), True)["allowed"]
    with pytest.raises(UserError):
        workspace.search(scene, None, "red roof", 8, 8, 0.5, None, True, Job("search"))


def test_suppression_removes_overlaps_and_caps_outputs():
    result = suppress([(0, 0, 30, 30), (1, 1, 31, 31), (60, 60, 90, 90)], np.array([0.9, 0.8, 0.7]), 0.75)
    assert len(result) == 1 and result[0]["score"] == 0.9
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert rect_dict((5, 6, 15, 26)) == dict(x=5, y=6, width=10, height=20)


def test_training_archive_roundtrip_and_unsafe_input(workspace):
    _, concept = teach(workspace)
    workspace.train(concept, Job("train"))
    data = workspace.export_training(concept)
    imported = workspace.import_training(data, Job("import"))["concept"]
    assert len(workspace.examples(imported)) == 4
    assert not next(item for item in workspace.concepts() if item["id"] == imported)["ready"]
    assert workspace.example_size(imported) == (64, 64)
    workspace.train(imported, Job("train"))
    before = len(workspace.concepts())
    for file in ("../outside.png", "/absolute.png", "C:\\outside.png"):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                "training.json",
                json.dumps(dict(format="atlas-training-v1", name="bad", examples=[dict(file=file, label=1)])),
            )
            archive.writestr(file, scene_bytes())
        with pytest.raises(UserError):
            workspace.import_training(output.getvalue(), Job("import"))
    assert len(workspace.concepts()) == before


def test_offline_api_full_workflow_and_network_restrictions(tmp_path):
    services = OfflineServices(tmp_path, engine=OfflineTestEngine())
    app = create_app(services=services, token="offline-test")
    with TestClient(app) as client:
        assert client.get("/api/state").status_code == 401
        client.get("/?token=offline-test", follow_redirects=False)
        state = client.get("/api/state")
        assert state.json()["offline"] is True
        assert "connect-src 'self'" in state.headers["Content-Security-Policy"]
        assert client.get("/api/state", headers={"Origin": "https://example.org"}).status_code == 403
        assert client.get("/api/state", headers={"Host": "example.org"}).status_code == 403
        imported = wait_job(
            client, client.post("/api/workspace/scenes?name=owned.png", content=scene_bytes()).json()
        )
        assert imported["status"] == "done"
        scene = imported["result"]["scene"]["id"]
        assert Image.open(io.BytesIO(client.get(f"/api/workspace/scenes/{scene}/full").content)).size == (
            256,
            192,
        )
        concept = client.post("/api/workspace/concepts", json={"name": "target"}).json()["id"]
        for x, y, label in [(16, 16, 1), (144, 16, 1), (16, 112, 0), (144, 112, 0)]:
            assert (
                client.post(
                    "/api/workspace/annotations",
                    json=dict(
                        concept=concept, scene=scene, bbox=dict(x=x, y=y, width=64, height=64), label=label
                    ),
                ).status_code
                == 200
            )
        assert (
            wait_job(client, client.post(f"/api/workspace/concepts/{concept}/train").json())["status"]
            == "done"
        )
        run = wait_job(
            client,
            client.post(
                "/api/workspace/search",
                json=dict(scene=scene, concept=concept, width=64, height=64, threshold=0.6),
            ).json(),
        )
        assert run["status"] == "done", run
        result = run["result"]
        assert result["detections"]
        prefix = "/api/workspace/runs/" + result["run"]
        csv = client.get(prefix + "/csv").content.decode("utf-8-sig")
        assert "x,y,width,height,model_score" in csv
        assert len(csv.strip().splitlines()) == len(result["detections"]) + 1
        assert Image.open(io.BytesIO(client.get(prefix + "/annotated").content)).size == (256, 192)
        assert client.post(prefix + "/feedback/0", json=dict(concept=concept, label=0)).status_code == 200
        assert not client.get("/api/state").json()["concepts"][0]["ready"]
        assert client.post(prefix + "/feedback/999", json=dict(concept=concept, label=0)).status_code == 400
        pack = client.get(f"/api/workspace/concepts/{concept}/export").content
        assert (
            wait_job(client, client.post("/api/workspace/training/import", content=pack).json())["status"]
            == "done"
        )
        assert client.post("/api/area", json={}).status_code == 404  # No online imagery API in this app.


def test_edits_blocked_during_training_and_cancelled_jobs(tmp_path):
    services = OfflineServices(tmp_path, engine=OfflineTestEngine())
    started, release = threading.Event(), threading.Event()

    def task(job):
        started.set()
        release.wait(5)
        job.check()

    with TestClient(create_app(services=services, token="test")) as client:
        client.get("/?token=test", follow_redirects=False)
        job = services.jobs.submit("train", task)
        assert started.wait(2)
        assert client.post("/api/workspace/concepts", json={"name": "blocked"}).status_code == 400
        client.post(f"/api/jobs/{job['id']}/cancel")
        release.set()
        assert wait_job(client, job)["status"] == "cancelled"
