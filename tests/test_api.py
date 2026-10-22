import io
import zipfile

import pytest
from conftest import archive, png, wait_job
from fastapi.testclient import TestClient

from atlas.server import create_app


def test_import_index_text_and_image_search_workflow(client):
    assert client.get("/api/state").json()["library"] == {"total": 0, "indexed": 0}
    imported = wait_job(client, client.post("/api/import", content=archive()).json())
    assert imported["status"] == "done"
    assert imported["result"]["imported"] == 2
    indexed = wait_job(client, client.post("/api/index").json())
    assert indexed["status"] == "done"
    assert indexed["result"]["indexed"] == 2
    text = wait_job(client, client.post("/api/search/text", json={"text": "سقف قرمز"}).json())
    assert text["result"]["buildings"][0]["title"] == "Red roof"
    image = wait_job(client, client.post("/api/search/image", content=png("blue")).json())
    result = image["result"]["buildings"][0]
    assert result["title"] == "Blue roof"
    assert result["score"] == 1.0
    response = client.get(result["image_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Temporary user queries do not accumulate on disk.
    assert not list(client.app.state.services.root.glob("tmp*.jpg"))
    repeated = wait_job(client, client.post("/api/index").json())
    assert repeated["result"]["indexed"] == 0


def test_api_requires_session_and_rejects_cross_origin(services):
    with TestClient(create_app(services=services, token="session")) as browser:
        assert browser.get("/api/state").status_code == 401
        response = browser.get("/?token=session", follow_redirects=False)
        assert response.status_code == 303
        assert "HttpOnly" in response.headers["set-cookie"]
        assert browser.get("/api/state").status_code == 200
        assert browser.post("/api/index", headers={"Origin": "https://unrelated.example"}).status_code == 403
        assert browser.get("/api/state", headers={"Host": "unrelated.example"}).status_code == 403


def test_search_requires_index_and_checks_inputs(client):
    assert client.post("/api/search/text", json={"text": "red roof"}).status_code == 400
    assert client.post("/api/search/text", json={"text": "  "}).status_code == 400
    assert client.post("/api/search/image", content=b"not-an-image").status_code == 400
    assert client.post("/api/search/image", content=b"").status_code == 400
    assert client.post("/api/area", json={"lat": 52, "lon": 4, "radius": 5000}).status_code == 422
    assert client.get("/api/images/missing").status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404


@pytest.mark.parametrize(
    "entries,files",
    [
        ([{"lat": 52, "lon": 4, "image": "../outside.png"}], {"../outside.png": png()}),
        ([{"lat": 95, "lon": 4, "image": "roof.png"}], {"roof.png": png()}),
        ([{"lat": "nan", "lon": 4, "image": "roof.png"}], {"roof.png": png()}),
        ([{"lat": 52, "lon": 4, "image": "missing.png"}], {}),
        ([{"lat": 52, "lon": 4, "image": "roof.png"}], {"roof.png": b"not-image"}),
        (
            [
                {"id": "same", "lat": 52, "lon": 4, "image": "a.png"},
                {"id": "same", "lat": 52, "lon": 4, "image": "b.png"},
            ],
            {"a.png": png(), "b.png": png()},
        ),
    ],
)
def test_invalid_import_never_changes_library(client, entries, files):
    result = wait_job(client, client.post("/api/import", content=archive(entries, files)).json())
    assert result["status"] == "failed"
    assert client.get("/api/state").json()["library"]["total"] == 0


def test_bad_zip_reports_actionable_error(client):
    result = wait_job(client, client.post("/api/import", content=b"not-a-zip").json())
    assert result["status"] == "failed"
    assert "ZIP" in result["error"]


def test_manifest_required(client):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as file:
        file.writestr("random.txt", "nothing")
    result = wait_job(client, client.post("/api/import", content=stream.getvalue()).json())
    assert result["status"] == "failed"
    assert "buildings.json" in result["error"]


def test_import_is_repeatable_without_duplicate_buildings(client):
    for _ in range(2):
        result = wait_job(client, client.post("/api/import", content=archive()).json())
        assert result["status"] == "done"
    assert client.get("/api/state").json()["library"]["total"] == 2
