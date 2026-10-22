import json

import httpx
import pytest
from conftest import png

from atlas.errors import UserError
from atlas.imagery import PDOK, TO_RD
from atlas.jobs import Job
from atlas.store import Store


def test_pdok_prepares_real_protocol_and_reuses_downloads(tmp_path):
    x, y = TO_RD.transform(4.8936, 52.3728)
    calls = []

    def handle(request):
        calls.append(request)
        parameters = request.url.params
        if parameters["request"] == "GetCapabilities":
            return httpx.Response(
                200, text="<WMS_Capabilities><Layer><Name>Actueel_orthoHR</Name></Layer></WMS_Capabilities>"
            )
        if parameters["request"] == "GetFeature":
            assert parameters["srsName"] == "EPSG:28992"
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [[x - 8, y - 8], [x + 8, y - 8], [x + 8, y + 8], [x - 8, y + 8], [x - 8, y - 8]]
                ],
            }
            return httpx.Response(
                200,
                json={
                    "features": [
                        {"id": "pand.123", "geometry": geometry, "properties": {"identificatie": "123"}}
                    ]
                },
            )
        assert parameters["request"] == "GetMap"
        assert parameters["layers"] == "Actueel_orthoHR"
        return httpx.Response(200, content=png(), headers={"Content-Type": "image/png"})

    provider = PDOK(httpx.Client(transport=httpx.MockTransport(handle)))
    store = Store(tmp_path)
    result = provider.prepare(store, Job("area"), 52.3728, 4.8936, 250, 30)
    assert result["downloaded"] == 1
    building = store.list()[0]
    assert building["lat"] == pytest.approx(52.3728, abs=1e-5)
    assert building["lon"] == pytest.approx(4.8936, abs=1e-5)
    assert building["geometry"]["type"] == "Polygon"
    assert store.image_path(building["id"]).is_file()
    assert json.loads(json.dumps(building))["source"].startswith("PDOK")
    repeated = provider.prepare(store, Job("area"), 52.3728, 4.8936, 250, 30)
    assert repeated["reused"] == 1
    assert sum(request.url.params["request"] == "GetMap" for request in calls) == 1


def test_provider_network_denial_is_actionable_and_preserves_library(tmp_path):
    provider = PDOK(httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403))))
    store = Store(tmp_path)
    with pytest.raises(UserError, match="403"):
        provider.prepare(store, Job("area"), 52.3728, 4.8936, 250, 30)
    assert store.counts()["total"] == 0


def test_unsupported_region_fails_before_network(tmp_path):
    def reject(request):
        raise AssertionError("Must not call a Netherlands provider outside coverage")

    provider = PDOK(httpx.Client(transport=httpx.MockTransport(reject)))
    with pytest.raises(UserError, match="هلند"):
        provider.prepare(Store(tmp_path), Job("area"), 35.7, 51.4, 250, 30)
