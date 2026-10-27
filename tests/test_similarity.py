import io
import json

import pytest
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine
from PIL import Image

from atlas.errors import Cancelled, UserError
from atlas.jobs import Job
from atlas.offline_server import OfflineServices, create_app
from atlas.similarity_validation import make_similarity_scene
from atlas.workshop import Workshop, box_tuple, intersection_over_union


def setup_scene(tmp_path):
    workshop = Workshop(tmp_path, OfflineTestEngine())
    picture, data, truth = make_similarity_scene()
    source = tmp_path / "scene.png"
    picture.save(source)
    scene = workshop.import_scene_file(source, source.name, Job("import"))["scene"]
    reference = workshop.similarity.add_file(data, "screenshot.jpg")
    return workshop, scene, reference, truth


def test_one_screenshot_finds_rotated_scaled_boundary_patterns(tmp_path):
    workshop, scene, reference, truth = setup_scene(tmp_path)
    options = dict(
        scene=scene["id"],
        reference=reference["id"],
        width=64,
        height=48,
        threshold=0.78,
        rotate=True,
        intelligent=True,
    )
    result = workshop.similarity.search(Job("similar"), **options)
    found = [box_tuple(x["bbox"], scene["width"], scene["height"]) for x in result["detections"]]
    assert all(any(intersection_over_union(b, target) > 0.5 for b in found) for target in truth), found
    assert len(found) <= 10
    assert not workshop.concepts()
    before = workshop.engine.calls
    result2 = workshop.similarity.search(Job("similar"), **{**options, "threshold": 0.79})
    assert workshop.engine.calls - before == 8  # only the reference, no scene re-embedding
    assert result2["total_results"] >= 3
    assert (
        Workshop(tmp_path, workshop.engine).similarity.reference(reference["id"])["name"] == "screenshot.jpg"
    )


def test_reference_from_box_and_exact_whole_scene_coverage(tmp_path):
    workshop, scene, _, _ = setup_scene(tmp_path)
    ref = workshop.similarity.add_box(scene["id"], dict(x=100, y=100, width=64, height=48))
    plan = workshop.similarity.plan(scene["id"], ref["id"], 8, 8)
    assert plan["allowed"] and plan["completed"] == 0
    assert not workshop.plan(scene["id"], 8, 8)["allowed"]  # legacy cap is exceeded
    run = workshop.similarity.search(
        Job("similar"),
        scene=scene["id"],
        reference=ref["id"],
        width=64,
        height=48,
        threshold=0.8,
        intelligent=False,
    )
    assert run["total_results"] >= 3
    assert workshop.similarity.plan(scene["id"], ref["id"], 64, 48, intelligent=False)["completed"] > 0
    with pytest.raises(UserError):
        workshop.similarity.search(
            Job("similar"),
            scene=scene["id"],
            reference=ref["id"],
            width=64,
            height=48,
            threshold=float("nan"),
        )


def test_scan_checkpoints_survive_cancellation_and_restart(tmp_path):
    workshop, scene, reference, _ = setup_scene(tmp_path)

    class Stop(Job):
        def update(self, progress, message):
            if "بخش 2 " in message:
                self.cancel_event.set()
            super().update(progress, message)

    options = dict(scene=scene["id"], reference=reference["id"], width=64, height=48, intelligent=False)
    with pytest.raises(Cancelled):
        workshop.similarity.search(Stop("similar"), **options)
    plan = workshop.similarity.plan(**options)
    assert plan["completed"] == 1
    resumed = Workshop(tmp_path, workshop.engine)
    run = resumed.similarity.search(Job("similar"), **options)
    assert run["total_results"] >= 3
    assert resumed.similarity.plan(**options)["completed"] == plan["windows"]


def test_results_pages_viewport_crops_and_streamed_export_are_not_capped_at_200(tmp_path):
    services = OfflineServices(tmp_path, OfflineTestEngine())
    workshop = services.workshop
    picture, data, truth = make_similarity_scene()
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    scene = workshop.import_scene(stream.getvalue(), "scene", Job("import"))["scene"]
    ref = workshop.similarity.add_file(data, "sample")
    run = workshop.similarity.search(
        Job("similar"), scene=scene["id"], reference=ref["id"], width=64, height=48, intelligent=False
    )
    # More than a page of accepted results, to exercise public paging/export routes.
    with workshop.connect() as db:
        db.execute(
            "DELETE FROM similar_bounds WHERE key IN (SELECT key FROM similar_detections WHERE run=?)",
            (run["run"],),
        )
        db.execute("DELETE FROM similar_detections WHERE run=?", (run["run"],))
        for n in range(451):
            x, y = (n % 25) * 64, (n // 25) * 24
            row = db.execute(
                "INSERT INTO similar_detections(run,idx,x,y,w,h,score,appearance,semantic,angle) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run["run"], n, x, y, 32, 16, 0.9, 0.9, 1, 0),
            )
            db.execute("INSERT INTO similar_bounds VALUES(?,?,?,?,?)", (row.lastrowid, x, x + 32, y, y + 16))
        settings = run["settings"]
        settings["total_results"] = 451
        db.execute("UPDATE runs SET settings=? WHERE id=?", (json.dumps(settings), run["run"]))
    with TestClient(create_app(services=services, token="similar-test")) as client:
        client.cookies.set("atlas_session", "similar-test")
        base = f"/api/workspace/runs/{run['run']}"
        first = client.get(base).json()
        assert first["total_results"] == 451 and len(first["detections"]) == 200
        last = client.get(base + "?offset=400").json()
        assert len(last["detections"]) == 51 and last["detections"][0]["index"] == 400
        assert client.get(base + "/crops/400").status_code == 200
        visible = client.post(base + "/visible", json=dict(x=0, y=0, width=200, height=100)).json()
        assert visible["detections"] and not visible["truncated"]
        assert all(d["bbox"]["x"] <= 200 and d["bbox"]["y"] <= 100 for d in visible["detections"])
        csv = client.get(base + "/csv").content.decode("utf-8-sig").splitlines()
        assert len(csv) == 452 and csv[-1].startswith("451,")
        annotated = Image.open(io.BytesIO(client.get(base + "/annotated").content))
        assert annotated.width == 1800
        assert (
            client.post(
                "/api/workspace/references/box",
                json=dict(scene=scene["id"], bbox=dict(x=100, y=100, width=64, height=48)),
            ).status_code
            == 200
        )
        assert client.get("/api/workspace/references").json()["references"]
