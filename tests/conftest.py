from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from atlas.server import Services, create_app


def png(color: str = "red") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, format="PNG")
    return stream.getvalue()


def archive(entries: list[dict] | None = None, files: dict[str, bytes] | None = None) -> bytes:
    stream = io.BytesIO()
    entries = (
        entries
        if entries is not None
        else [
            {"id": "red", "lat": 52.372, "lon": 4.893, "image": "red.png", "title": "Red roof"},
            {"id": "blue", "lat": 52.373, "lon": 4.894, "image": "blue.png", "title": "Blue roof"},
        ]
    )
    with zipfile.ZipFile(stream, "w") as file:
        file.writestr("buildings.json", json.dumps({"buildings": entries}))
        for name, data in (
            files if files is not None else {"red.png": png(), "blue.png": png("blue")}
        ).items():
            file.writestr(name, data)
    return stream.getvalue()


class TestEngine:
    """Deterministic test double for plumbing tests; not an AI accuracy test."""

    __test__ = False

    @staticmethod
    def status():
        return {"image_loaded": True, "text_loaded": True, "device": "test-double"}

    @staticmethod
    def prepare():
        pass

    @staticmethod
    def vector(red: bool):
        vector = np.zeros(512, dtype=np.float32)
        vector[0 if red else 1] = 1
        return vector

    def image(self, path):
        with Image.open(path) as image:
            pixel = image.convert("RGB").getpixel((0, 0))
        return self.vector(pixel[0] > pixel[2])

    def text(self, text):
        return self.vector("قرمز" in text or "red" in text.lower())


def wait_job(client: TestClient, job: dict) -> dict:
    for _ in range(300):
        result = client.get(f"/api/jobs/{job['id']}").json()
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


@pytest.fixture
def services(tmp_path):
    return Services(tmp_path, engine=TestEngine())


@pytest.fixture
def client(services):
    with TestClient(create_app(services=services, token="test-session")) as client:
        client.get("/?token=test-session", follow_redirects=False)
        yield client
