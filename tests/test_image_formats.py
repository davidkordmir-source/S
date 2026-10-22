import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from offline_support import OfflineTestEngine
from PIL import Image

from atlas import ecw_reader
from atlas.errors import UserError
from atlas.image_io import format_support, read_picture
from atlas.jobs import Job
from atlas.offline_server import OfflineServices, create_app
from atlas.workshop import Workshop

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "formats/manifest.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["file"])
def test_real_image_files_import_as_scene_and_example(tmp_path, case):
    data = (DATA / "formats" / case["file"]).read_bytes()
    picture = read_picture(data, scene=True)
    assert picture.mode == "RGB" and picture.size == tuple(case["size"])
    if case["file"] != "alpha.png":
        assert np.asarray(picture).std() > 5
    workshop = Workshop(tmp_path, OfflineTestEngine())
    scene = workshop.import_scene(data, case["file"], Job("import"))["scene"]
    assert [scene["width"], scene["height"]] == case["size"]
    concept = workshop.new_concept("target")["id"]
    example = workshop.add_file(concept, data, 1)
    with Image.open(workshop.example_path(example["id"])) as stored:
        assert stored.mode == "RGB"


def test_first_frame_and_psd_composite_policy_is_explicit():
    for name in ("animated.gif", "multipage.tiff"):
        image = read_picture((DATA / "formats" / name).read_bytes())
        assert image.getpixel((5, 5))[0] > 180
        assert "اول" in " ".join(image.info["atlas_import_notes"])
    psd = read_picture((DATA / "formats/merged.psd").read_bytes())
    assert psd.getpixel((5, 5)) == (208, 48, 32)
    assert "ادغام" in " ".join(psd.info["atlas_import_notes"])


def test_transparency_gray16_cmyk_and_exif_orientation():
    alpha = read_picture((DATA / "formats/alpha.png").read_bytes())
    assert alpha.getpixel((0, 0)) == (255, 155, 155)
    gray = read_picture((DATA / "formats/gray16.tiff").read_bytes())
    assert np.asarray(gray).max() > 240 and np.asarray(gray).min() < 10
    assert gray.info["atlas_import_notes"]
    cmyk = read_picture((DATA / "formats/camera-cmyk.jpg").read_bytes())
    assert cmyk.getpixel((5, 5))[0] > 180
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20), "red").save(output, format="JPEG", exif=exif)
    assert read_picture(output.getvalue()).size == (20, 40)


def test_broken_files_and_document_formats_are_rejected():
    for data in (b"not an image", b"%PDF-1.7\n", b"GIF89a", b"\x89PNG\r\n\x1a\n"):
        with pytest.raises(UserError):
            read_picture(data)
    # EPS must not invoke an external Ghostscript program.
    with pytest.raises(UserError):
        read_picture(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 20 20\n")


def test_picker_manifest_and_ecw_detection():
    formats = format_support()
    for suffix in (".ecw", ".jp2", ".heic", ".avif", ".psd", ".gif", ".img", ".ntf", ".pix"):
        assert suffix in formats["accept"].split(",")
    for name in ("jrc.ecw", "stefan_full_rgba_ecwv3_meta.ecw"):
        assert ecw_reader.is_ecw((DATA / "ecw" / name).read_bytes())
    assert not ecw_reader.is_ecw(b"not an ecw")


@pytest.mark.parametrize("label", (0, 1))
def test_upload_example_api_accepts_numeric_query_labels(tmp_path, label):
    services = OfflineServices(tmp_path, engine=OfflineTestEngine())
    with TestClient(create_app(services=services, token="format-test")) as client:
        client.get("/?token=format-test", follow_redirects=False)
        concept = client.post("/api/workspace/concepts", json={"name": "format target"}).json()["id"]
        response = client.post(
            f"/api/workspace/concepts/{concept}/examples?label={label}",
            content=(DATA / "formats/basic.jp2").read_bytes(),
        )
        assert response.status_code == 200, response.text
        assert response.json()["label"] == label
        assert (
            client.post(
                f"/api/workspace/concepts/{concept}/examples?label=2",
                content=(DATA / "formats/basic.heic").read_bytes(),
            ).status_code
            == 422
        )


@pytest.mark.skipif(not ecw_reader.available(), reason="Independent ECW reader is required")
@pytest.mark.parametrize("name", ("jrc.ecw", "stefan_full_rgba_ecwv3_meta.ecw"))
def test_actual_ecw_versions_import_and_obey_dimensions(tmp_path, name):
    data = (DATA / "ecw" / name).read_bytes()
    picture = read_picture(data, scene=True)
    assert picture.mode == "RGB" and min(picture.size) > 100
    assert np.asarray(picture).std() > 5
    if name == "jrc.ecw":
        red = np.asarray(picture)[:, :, 0]
        # GDAL's reference SDK-v5 statistics for this upstream fixture.
        assert abs(red.mean() - 141.606) < 0.5 and abs(red.std() - 67.2919) < 0.5
    with pytest.raises(UserError, match="ابعاد"):
        ecw_reader.read_ecw(data, 100)
    workspace = Workshop(tmp_path, OfflineTestEngine())
    scene = workspace.import_scene(data, name, Job("import"))["scene"]
    assert (scene["width"], scene["height"]) == picture.size
    concept = workspace.new_concept("ECW")["id"]
    assert workspace.add_file(concept, data, 1)["id"]
