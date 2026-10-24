from __future__ import annotations

import io
import ipaddress
import json
import logging
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.special import expit

from .engine import Engine
from .jobs import Job
from .workshop import Workshop


def validate_packaged_formats(root: Path) -> dict:
    from importlib.metadata import distribution

    from .ecw_reader import available
    from .image_io import read_picture
    from .raster_reader import bundle_directory
    from .scene_source import SceneReader

    data = (
        Path(sys._MEIPASS) / "validation-data"
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2] / "tests/data"
    )
    root.mkdir(parents=True, exist_ok=True)
    files = []
    with deny_external_network() as attempts:
        assert "MIT License" in (distribution("ecw2tiff").read_text("licenses/LICENSE") or "")
        assert not (bundle_directory() / "bin/NCSEcw.dll").exists()
        assert not list(bundle_directory().rglob("*ECW*.dll"))
        for entry in json.loads((data / "formats/manifest.json").read_text()):
            logging.info("Checking image decoder: %s", entry["file"])
            image = read_picture((data / "formats" / entry["file"]).read_bytes(), scene=True)
            assert image.mode == "RGB" and image.size == tuple(entry["size"]), entry["file"]
            if entry["file"] != "alpha.png":
                assert np.asarray(image).std() > 5, entry["file"]
            files.append(entry["file"])
        assert available(), "The Windows package is missing its independent ECW decoder"
        for name in ("jrc.ecw", "stefan_full_rgba_ecwv3_meta.ecw"):
            image = read_picture((data / "ecw" / name).read_bytes(), scene=True)
            assert image.mode == "RGB" and min(image.size) > 100 and np.asarray(image).std() > 5
            if name == "jrc.ecw":
                red = np.asarray(image)[:, :, 0]
                assert abs(red.mean() - 141.606) < 0.5 and abs(red.std() - 67.2919) < 0.5
            files.append(name)
        for name in json.loads((data / "rasters/source.json").read_text())["files"]:
            image = read_picture((data / "rasters" / name).read_bytes(), scene=True)
            assert image.mode == "RGB" and min(image.size) > 0 and np.asarray(image).size > 0
            files.append(name)
        # Validate the actual file-backed path as well as bounded sample decoding.
        for filename in files:
            group = (
                "ecw"
                if filename.endswith(".ecw")
                else "formats"
                if (data / "formats" / filename).exists()
                else "rasters"
            )
            with SceneReader(data / group / filename) as reader:
                image = reader.read(maximum=1024)
                assert image.mode == "RGB" and min(image.size) > 0
        assert not attempts
        report = dict(
            ecw_decoder="ecw2tiff 0.1.1 (MIT)",
            proprietary_ecw_sdk=False,
            status="passed",
            offline=True,
            external_python_connections=len(attempts),
            files=files,
            ecw_versions=[2, 3],
        )
    (root / "formats-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Packaged image format validation passed: %s", report)
    return report


@contextmanager
def deny_external_network():
    """Fail on outbound socket attempts, including during initial model loading."""
    original_connect = socket.socket.connect
    original_sendto = socket.socket.sendto
    attempts = []

    def check(address):
        if not isinstance(address, tuple):
            return
        try:
            local = ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            local = address[0] == "localhost"
        if not local:
            attempts.append(str(address[0]))
            raise RuntimeError("Offline validation blocked an external network attempt")

    def connect(instance, address):
        check(address)
        return original_connect(instance, address)

    def sendto(instance, data, *args):
        check(args[-1])
        return original_sendto(instance, data, *args)

    socket.socket.connect, socket.socket.sendto = connect, sendto
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.sendto = original_connect, original_sendto


def fixture(held_out=False):
    """Controlled functional fixture, not a satellite detection benchmark."""
    picture = Image.new("RGB", (256, 192), "#687a60")
    draw = ImageDraw.Draw(picture)
    entries = [(16, 16, 1), (144, 16, 1), (16, 112, 0), (144, 112, 0)]
    if held_out:
        entries = [(32, 64, 1), (160, 64, 0)]
    for x, y, label in entries:
        draw.rectangle((x, y, x + 63, y + 63), fill="#c6bca2")
        color = ("#c23c2b" if held_out else "#d13a29") if label else "#3564bd"
        draw.polygon([(x + 5, y + 10), (x + 54, y + 5), (x + 59, y + 50), (x + 10, y + 56)], fill=color)
        draw.line((x + 6, y + 12, x + 56, y + 48), fill="#e48a68" if label else "#638fcb", width=2)
    output = io.BytesIO()
    picture.save(output, "PNG")
    return output.getvalue(), entries


def validate_packaged_ai(root: Path) -> dict:
    """Exercise bundled models, local learning and search with all outbound sockets blocked."""
    root.mkdir(parents=True, exist_ok=True)
    with deny_external_network() as attempts:
        engine = Engine(root)
        logging.info("Offline AI check: loading the two bundled models without network access")
        engine.prepare()
        with tempfile.TemporaryDirectory(prefix="atlas-offline-validation-", dir=root) as directory:
            workshop = Workshop(Path(directory), engine)
            data, entries = fixture()
            training = workshop.import_scene(data, "training.png", Job("import"))["scene"]["id"]
            concept = workshop.new_concept("ساختمان با سقف قرمز")["id"]
            for x, y, label in entries:
                workshop.add_box(concept, training, dict(x=x, y=y, width=64, height=64), label)
            workshop.train(concept, Job("train"))
            assert workshop.concepts()[0]["ready"]
            data, held_out = fixture(True)
            scene = workshop.import_scene(data, "held-out-user-image.png", Job("import"))["scene"]["id"]
            with Image.open(io.BytesIO(data)) as image:
                vectors = engine.images([image.crop((x, y, x + 64, y + 64)) for x, y, _ in held_out])
            model = json.loads(workshop.concept(concept)["model"])
            scores = expit(vectors @ np.asarray(model["weights"]) + model["intercept"])
            assert scores[0] > scores[1] + 0.1, scores.tolist()
            result = workshop.search(scene, concept, "", 64, 64, 0.55, None, True, Job("search"))
            assert result["detections"], "The learned target was not found in the second image"
            assert any(
                item["bbox"]["x"] <= 64 <= item["bbox"]["x"] + item["bbox"]["width"]
                and item["bbox"]["y"] <= 96 <= item["bbox"]["y"] + item["bbox"]["height"]
                for item in result["detections"]
            )
            for item in result["detections"]:
                box = item["bbox"]
                assert 0 <= box["x"] < box["x"] + box["width"] <= 256
                assert 0 <= box["y"] < box["y"] + box["height"] <= 192
            workshop.add_box(concept, scene, result["detections"][0]["bbox"], 1)
            assert not workshop.concepts()[0]["ready"]
            workshop.train(concept, Job("retrain"))
            assert Workshop(Path(directory), engine).concepts()[0]["ready"]
            exported = workshop.export_training(concept)
            restored = workshop.import_training(exported, Job("import"))["concept"]
            assert len(workshop.examples(restored)) == 5
            persian, english = engine.text("ساختمانی با سقف قرمز"), engine.text("a building with a red roof")
            alignment = float(np.dot(persian, english))
            assert alignment > 0.5
            text_result = workshop.search(
                scene, None, "ساختمانی با سقف قرمز", 64, 64, 0.5, None, True, Job("text-search")
            )
            assert text_result["detections"]
            assert attempts == [], attempts
            report = dict(
                status="passed",
                offline=True,
                external_network_attempts=len(attempts),
                training_positive=2,
                training_negative=2,
                held_out_positive_score=round(float(scores[0]), 5),
                held_out_negative_score=round(float(scores[1]), 5),
                windows=result["settings"]["windows"],
                detections=len(result["detections"]),
                feedback_retraining=True,
                training_pack_roundtrip=True,
                persian_english_alignment=round(alignment, 5),
            )
    (root / "ai-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Offline packaged AI check passed: %s", report)
    return report
