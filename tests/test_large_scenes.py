import io
import json
import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine, scene_bytes
from PIL import Image

from atlas.errors import Cancelled, UserError
from atlas.jobs import Job
from atlas.large_validation import FAR_X, FAR_Y, TIB, make_sparse_scene, mark_sparse, validate_large_files
from atlas.offline_server import OfflineServices, create_app
from atlas.scene_source import SceneReader
from atlas.workshop import Workshop


def test_sparse_terabyte_scene_has_real_far_tile_and_bounded_memory(tmp_path):
    report = validate_large_files(tmp_path)
    assert report["logical_file_bytes"] > TIB
    assert report["real_tile_offset"] > TIB
    if report["allocated_bytes"] is not None:
        assert report["allocated_bytes"] < 10 * 1024 * 1024


def test_large_raster_training_roi_search_exports_and_persistence(tmp_path):
    source = tmp_path / "satellite.tif"
    make_sparse_scene(source, 200 * 1024 * 1024)
    services = OfflineServices(tmp_path / "data", OfflineTestEngine())
    workshop = services.workshop
    scene = workshop.import_scene_file(source, source.name, Job("import"))["scene"]["id"]
    concept = workshop.new_concept("red roof")["id"]
    for x, y, label in [(FAR_X, FAR_Y, 1), (FAR_X + 128, FAR_Y, 1), (0, 0, 0), (128, 0, 0)]:
        workshop.add_box(concept, scene, dict(x=x, y=y, width=128, height=128), label)
    workshop.train(concept, Job("train"))
    result = workshop.search(
        scene, concept, "", 128, 128, 0.5, dict(x=FAR_X, y=FAR_Y, width=256, height=256), False, Job("scan")
    )
    assert result["detections"]
    assert all(d["bbox"]["x"] >= FAR_X and d["bbox"]["y"] >= FAR_Y for d in result["detections"])
    assert max(workshop.crop_detection(result["run"], 0).size) <= 1024
    with TestClient(create_app(token="large-test", services=services)) as client:
        client.cookies.set("atlas_session", "large-test")
        base = f"/api/workspace/scenes/{scene}"
        tile = client.get(base + f"/tiles/0/{FAR_X // 512}/{FAR_Y // 512}")
        assert tile.status_code == 200
        assert Image.open(io.BytesIO(tile.content)).size == (512, 512)
        assert client.get(base + "/tiles/0/999/0").status_code == 400
        assert client.get(base + "/tiles/-1/0/0").status_code == 400
        prefix = f"/api/workspace/runs/{result['run']}"
        csv = client.get(prefix + "/csv").content.decode("utf-8-sig")
        assert str(FAR_X) in csv and str(FAR_Y) in csv
        picture = Image.open(io.BytesIO(client.get(prefix + "/annotated").content))
        assert max(picture.size) == 4096
        assert json.loads(picture.info["atlas_original_size"]) == [30000, 24000]
        assert Image.open(io.BytesIO(client.get(base + "/full").content)).width <= 2400
    reopened = Workshop(tmp_path / "data", OfflineTestEngine())
    assert reopened.scene_path(scene) == source
    assert reopened.concepts()[0]["ready"]
    # Cached search features and tiles must not hide a changed source.
    with source.open("r+b") as file:
        file.truncate(source.stat().st_size + 1)
    with pytest.raises(UserError, match="تغییر"):
        reopened.tile(scene, 0, 0, 0)
    with pytest.raises(UserError, match="تغییر"):
        reopened.search(scene, concept, "", 128, 128, 0.5, result["settings"]["scope"], False, Job("scan"))


def test_link_copy_cancellation_and_legacy_scene_migration(tmp_path):
    source = tmp_path / "original.png"
    source.write_bytes(scene_bytes())
    workshop = Workshop(tmp_path / "data", OfflineTestEngine())
    linked = workshop.import_scene_file(source, source.name, Job("import"))["scene"]
    copied = workshop.import_scene_file(source, source.name, Job("import"), copy=True)["scene"]
    assert linked["storage"] == "linked" and copied["storage"] == "managed"
    assert workshop.scene_path(copied["id"]) != source
    cancelled = Job("import")
    cancelled.cancel_event.set()
    before = set(workshop.images.iterdir())
    with pytest.raises(Cancelled):
        workshop.import_scene_file(source, source.name, cancelled, copy=True)
    assert set(workshop.images.iterdir()) == before
    assert source.exists()
    source.unlink()
    with pytest.raises(UserError, match="فایل اصلی"):
        workshop.scene_path(linked["id"])
    with workshop.reader(copied["id"]) as reader:
        assert reader.read().size == (256, 192)
    # Simulate the pre-file-backed database rows, which refer to managed PNGs.
    legacy = workshop.images / "legacy.png"
    legacy.write_bytes(scene_bytes())
    with workshop.connect() as db:
        db.execute("INSERT INTO scenes(id,name,width,height,created) VALUES('legacy','old',256,192,0)")
    assert Workshop(workshop.root, workshop.engine).reader("legacy").read().size == (256, 192)


def test_ecw_file_over_one_tib_is_read_by_path_not_bytes(tmp_path, monkeypatch):
    source = tmp_path / "huge.ecw"
    shutil.copyfile(Path(__file__).parent / "data/ecw/jrc.ecw", source)
    with source.open("r+b") as file:
        mark_sparse(file)
        file.truncate(TIB + 1)

    def no_bytes(*args, **kwargs):
        raise AssertionError("A large source must not be read into bytes")

    monkeypatch.setattr(Path, "read_bytes", no_bytes)
    with SceneReader(source) as reader:
        assert reader.read((0, 0, 100, 100)).size == (100, 100)
        assert reader.read((0, 0, 100, 100)).getextrema() != ((0, 0),) * 3


def test_streamed_http_scene_upload_exceeds_old_150_mib_limit(tmp_path):
    source = tmp_path / "large.ecw"
    shutil.copyfile(Path(__file__).parent / "data/ecw/jrc.ecw", source)
    with source.open("r+b") as file:
        mark_sparse(file)
        file.truncate(160 * 1024 * 1024)
    services = OfflineServices(tmp_path / "data", OfflineTestEngine())
    app = create_app(token="stream-test", services=services)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", cookies={"atlas_session": "stream-test"}, timeout=60
        ) as client:
            with source.open("rb") as file:
                response = client.post(
                    "/api/workspace/scenes?name=large.ecw", content=iter(lambda: file.read(1024 * 1024), b"")
                )
            assert response.status_code == 200, response.text
            job = response.json()
            for _ in range(100):
                job = client.get(f"/api/jobs/{job['id']}").json()
                if job["status"] in ("done", "failed"):
                    break
                time.sleep(0.05)
            assert job["status"] == "done", job
            assert job["result"]["scene"]["source_size"] == 160 * 1024 * 1024
            assert job["result"]["scene"]["storage"] == "managed"
    finally:
        server.should_exit = True
        thread.join(5)
        assert not thread.is_alive()


def test_desktop_bridge_uses_selected_path_without_upload(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from atlas.desktop_bridge import FileBridge

    source = tmp_path / "picked.png"
    source.write_bytes(scene_bytes())
    services = OfflineServices(tmp_path / "data", OfflineTestEngine())
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    bridge = FileBridge(services, None)
    try:
        selection = bridge.chooseScene(False)
        for _ in range(100):
            job = services.jobs.get(selection["job"]["id"]).view()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert job["status"] == "done", job
        scene = job["result"]["scene"]
        assert scene["source_path"] == str(source) and scene["storage"] == "linked"
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
        assert bridge.chooseScene(False) == {"cancelled": True}
    finally:
        services.close()


def test_huge_ecw_index_is_rejected_before_native_allocation(tmp_path):
    import struct

    from atlas.ecw_reader import check_scene_index_budget

    source = tmp_path / "huge-index.ecw"
    header = bytearray(89)
    header[:6] = bytes([101, 2, 1, 1, 1, 4])
    header[14:22] = struct.pack(">HHHH", 3, 1, 64, 64)
    level = bytes([0]) + struct.pack(">IIIII", 1_000_000, 1_000_000, 1, 1, 1)
    source.write_bytes(header + level + bytes(4))
    with pytest.raises(UserError, match="جدول بلوک"):
        check_scene_index_budget(source)


def test_raster_block_cache_stays_bounded_after_reader_closes():
    import rasterio

    original = rasterio.env.get_gdal_config("GDAL_CACHEMAX")
    try:
        rasterio.env.set_gdal_config("GDAL_CACHEMAX", 256 * 1024 * 1024)
        with SceneReader(Path(__file__).parent / "data/formats/basic.tiff", {"kind": "rasterio"}) as reader:
            assert reader.read().mode == "RGB"
        # Closing a reader must not restore a larger process-wide cache while
        # other scene/tile readers are still active.
        assert rasterio.env.get_gdal_config("GDAL_CACHEMAX") == 64 * 1024 * 1024
    finally:
        rasterio.env.set_gdal_config("GDAL_CACHEMAX", original)
