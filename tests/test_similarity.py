import io
import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine
from PIL import Image

from atlas.errors import Cancelled, UserError
from atlas.jobs import Job
from atlas.offline_server import OfflineServices, create_app
from atlas.reference_matcher import ReferenceMatcher
from atlas.similarity_validation import make_similarity_scene
from atlas.visual_features import appearance_scores, dense_proposals
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
    assert workshop.engine.calls - before == 0  # persisted reference and scene features are reused
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
                (run["run"], n, x, y, 32, 16, 0.9 if n < 150 else 0.75 if n < 300 else 0.6, 0.9, 1, 0),
            )
            db.execute("INSERT INTO similar_bounds VALUES(?,?,?,?,?)", (row.lastrowid, x, x + 32, y, y + 16))
        settings = run["settings"]
        settings["total_results"] = 451
        settings["level_counts"] = dict(high=150, medium=150, review=151)
        db.execute("UPDATE runs SET settings=? WHERE id=?", (json.dumps(settings), run["run"]))
    with TestClient(create_app(services=services, token="similar-test")) as client:
        client.cookies.set("atlas_session", "similar-test")
        base = f"/api/workspace/runs/{run['run']}"
        first = client.get(base).json()
        assert first["total_results"] == 451 and len(first["detections"]) == 200
        assert first["level_counts"] == dict(high=150, medium=150, review=151)
        for level, count in [("high", 150), ("medium", 150), ("review", 151)]:
            filtered = client.get(base + f"?level={level}").json()
            assert filtered["filtered_results"] == count
            assert all(d["level"] == level for d in filtered["detections"])
            assert filtered["detections"][0]["similarity_percent"] == (
                90 if level == "high" else 75 if level == "medium" else 60
            )
            visible_level = client.post(
                base + f"/visible?level={level}", json=dict(x=0, y=0, width=1800, height=560)
            ).json()
            assert len(visible_level["detections"]) == count
            assert all(d["level"] == level for d in visible_level["detections"])
        focus = client.get(base + "?level=review&focus=450").json()
        assert focus["offset"] == 0 and focus["detections"][-1]["index"] == 450
        last = client.get(base + "?offset=400").json()
        assert len(last["detections"]) == 51 and last["detections"][0]["index"] == 400
        assert client.get(base + "/crops/400").status_code == 200
        visible = client.post(base + "/visible", json=dict(x=0, y=0, width=200, height=100)).json()
        assert visible["detections"] and not visible["truncated"]
        assert all(d["bbox"]["x"] <= 200 and d["bbox"]["y"] <= 100 for d in visible["detections"])
        csv = client.get(base + "/csv").content.decode("utf-8-sig").splitlines()
        assert len(csv) == 452 and csv[-1].startswith("451,")
        assert (
            "similarity_percent,level,pixel_similarity,structure_similarity,detail_similarity,semantic_similarity"
            in csv[0]
        )
        assert ",60.0,review," in csv[-1]
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


def test_dense_feature_proposals_do_not_require_any_pixel_template_peak():
    class Features:
        def details(self, pictures):
            assert all(max(p.size) <= 448 for p in pictures)
            grid = np.zeros((32, 32, 2), np.float32)
            grid[:, :, 0] = 1
            grid[7:15, 10:18] = [0, 1]
            return [dict(grid=grid, cls=np.array([0, 1]))]

    # The pixels contain no target at all: only the independent feature map does.
    proposals = list(
        dense_proposals(
            Image.new("RGB", (448, 448), "gray"),
            Features(),
            [
                dict(
                    pooled=np.array([0, 1], np.float32), spatial=np.tile(np.array([0, 1], np.float32), (9, 1))
                )
            ],
            112,
            112,
            (1,),
            False,
            lambda: None,
        )
    )
    assert any(intersection_over_union(box, (140, 98, 252, 210)) > 0.9 for box, score in proposals)


def test_flat_reference_does_not_make_every_edge_match_perfectly():
    sample = Image.new("RGB", (64, 48), "red")
    crop = Image.fromarray(np.random.default_rng(42).integers(0, 255, (48, 64, 3), dtype=np.uint8))
    rgb, edge, _ = appearance_scores(crop, sample, (0,))
    assert rgb < 0.5 and edge == 0


def test_similarity_schema_upgrade_preserves_existing_rows(tmp_path):
    workshop, scene, ref, _ = setup_scene(tmp_path)
    path = workshop.folder / "workspace.sqlite3"
    with sqlite3.connect(path) as db:
        for table in ("similar_candidates", "similar_detections"):
            db.execute(f"ALTER TABLE {table} DROP COLUMN detail")
            db.execute(f"ALTER TABLE {table} DROP COLUMN structure")
    restored = Workshop(tmp_path, workshop.engine)
    assert restored.similarity.reference(ref["id"])["id"] == ref["id"]
    assert restored.scene(scene["id"])["id"] == scene["id"]
    with restored.connect() as db:
        assert {"detail", "structure"} <= {r[1] for r in db.execute("PRAGMA table_info(similar_candidates)")}


def test_optional_multiview_library_and_reference_feature_cache(tmp_path):
    workshop, scene, ref, _ = setup_scene(tmp_path)
    library = workshop.similarity.library
    target = library.create("ساختمان از چند زاویه")
    library.add(target["id"], ref["id"])
    settings = workshop.similarity.plan(scene["id"], group=target["id"], width=64, height=48)["settings"]
    assert settings["reference_count"] == 1 and not settings["colour_evidence"]
    ReferenceMatcher(workshop.similarity, settings, Job("prepare"))
    before = workshop.engine.calls
    ReferenceMatcher(workshop.similarity, settings, Job("reuse"))
    assert workshop.engine.calls == before
    second = workshop.similarity.add_box(scene["id"], dict(x=752, y=170, width=48, height=64))
    library.add(target["id"], second["id"])
    more = workshop.similarity.plan(scene["id"], group=target["id"], width=64, height=48)
    assert more["settings"]["reference_count"] == 2
    assert (
        more["settings"]["references"][1]["width"] == 48 and more["settings"]["references"][1]["height"] == 64
    )
    ReferenceMatcher(workshop.similarity, more["settings"], Job("new-view"))
    assert workshop.engine.calls - before == 8
    assert more["scan"] != workshop.similarity.plan(scene["id"], ref["id"], 64, 48)["scan"]
    library.change(target["id"], second["id"], enabled=False)
    assert len(Workshop(tmp_path, workshop.engine).similarity.library.group(target["id"])["references"]) == 2
    assert workshop.similarity.plan(scene["id"], group=target["id"])["settings"]["reference_count"] == 1
    library.change(target["id"], ref["id"], enabled=False)
    with pytest.raises(UserError, match="حداقل یک"):
        workshop.similarity.plan(scene["id"], group=target["id"])


def test_group_api_single_optional_many_duplicates_and_removal(tmp_path):
    services = OfflineServices(tmp_path, OfflineTestEngine())
    _, data, _ = make_similarity_scene()
    with TestClient(create_app(services=services, token="profiles")) as client:
        client.cookies.set("atlas_session", "profiles")
        first = client.post("/api/workspace/references?name=first.jpg", content=data).json()
        assert first["group"]
        id = first["group"]
        base = "/api/workspace/reference-groups/" + id
        group = client.get(base).json()
        assert len(group["references"]) == 1 and group["references"][0]["enabled"]
        duplicate = client.post(
            "/api/workspace/references?name=duplicate.jpg&group=" + id, content=data
        ).json()
        assert duplicate["id"] == first["id"]
        assert len(client.get(base).json()["references"]) == 1
        assert client.patch(base, json={"name": "هدف جدید"}).json()["name"] == "هدف جدید"
        assert not client.patch(base + "/members/" + first["id"], json={"enabled": False}).json()[
            "references"
        ][0]["enabled"]
        assert client.delete(base + "/members/" + first["id"]).json()["references"] == []
        # Removing a view from teaching cannot break previously saved run images.
        assert client.get("/api/workspace/references/" + first["id"] + "/image").status_code == 200


def test_colour_only_uniform_reference_cannot_start_a_search(tmp_path):
    workshop, scene, _, _ = setup_scene(tmp_path)
    ref = workshop.similarity.add_reference(Image.new("RGB", (64, 48), "red"), "colour-only.png")
    with pytest.raises(UserError, match="رنگ یکنواخت"):
        workshop.similarity.search(Job("colour"), scene=scene["id"], reference=ref["id"], width=64, height=48)


def test_old_independent_references_migrate_without_merging_and_deduplicate(tmp_path):
    workshop, scene, ref, _ = setup_scene(tmp_path)
    other = workshop.similarity.add_box(scene["id"], dict(x=752, y=170, width=48, height=64))
    with workshop.connect() as db:
        db.execute("DELETE FROM reference_members")
        db.execute("DELETE FROM reference_groups")
        db.execute("DELETE FROM similarity_meta WHERE key='groups-v1'")
    restored = Workshop(tmp_path, workshop.engine).similarity
    groups = restored.library.groups()
    assert len(groups) == 2 and all(g["total"] == 1 for g in groups)
    original = next(g for g in groups if restored.library.group(g["id"])["references"][0]["id"] == ref["id"])
    copy = restored.add_file((restored.folder / f"{ref['id']}.png").read_bytes(), "copy.png")
    assert restored.library.add(original["id"], copy["id"])["id"] == ref["id"]
    assert len(restored.library.group(original["id"])["references"]) == 1
    assert restored.reference(other["id"])["id"] == other["id"]


def test_corrupt_reference_feature_cache_is_rebuilt(tmp_path):
    workshop, scene, ref, _ = setup_scene(tmp_path)
    settings = workshop.similarity.plan(scene["id"], ref["id"], 64, 48)["settings"]
    ReferenceMatcher(workshop.similarity, settings, Job("prepare"))
    next((workshop.similarity.folder / "features").glob("*.npz")).write_bytes(b"interrupted cache write")
    before = workshop.engine.calls
    ReferenceMatcher(workshop.similarity, settings, Job("recover"))
    assert workshop.engine.calls - before == 8


def test_uniform_candidate_cannot_inherit_spurious_normalized_edge_score(tmp_path):
    workshop, scene, reference, _ = setup_scene(tmp_path)
    settings = workshop.similarity.plan(
        scene["id"],
        reference["id"],
        width=64,
        height=48,
        intelligent=False,
        geometry=False,
    )["settings"]
    matcher = ReferenceMatcher(workshop.similarity, settings, Job("match"))
    # A normalized OpenCV response may be arbitrarily high at zero variance.
    # Reproduce that proposal independently of the host's OpenCV/CPU math.
    chunk = [((10, 20, 74, 68), 0.99, 0, 0, 0.0, 0.99)]
    assert matcher.score(chunk, [Image.new("RGB", (64, 48), "#647b60")], "scan", 0) == []
    with Image.open(workshop.similarity.folder / f"{reference['id']}.png") as source:
        textured = source.convert("RGB")
    normal = ((90, 20, 154, 68), 0.99, 0, 0, 0.9, 0.9)
    records = matcher.score(chunk + [normal], [Image.new("RGB", (64, 48), "red"), textured], "scan", 0)
    assert len(records) == 1 and records[0][2:4] == (90, 20)
