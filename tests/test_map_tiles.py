import httpx
import pytest
from conftest import png

from atlas.errors import UserError
from atlas.imagery import PDOK
from atlas.map_tiles import MapTiles


def test_street_tiles_use_fixed_host_and_honor_freshness(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == "https://tile.openstreetmap.org/16/33658/21540.png"
        return httpx.Response(
            200, content=png(), headers={"Content-Type": "image/png", "Cache-Control": "max-age=600"}
        )

    provider = PDOK(httpx.Client(transport=httpx.MockTransport(handle)))
    tiles = MapTiles(tmp_path, provider)
    first = tiles.get("street", 16, 33658, 21540)
    second = tiles.get("street", 16, 33658, 21540)
    assert first[0] == second[0]
    assert first[1] == second[1] == "image/png"
    assert 0 < second[2] <= first[2] == 600
    assert len(requests) == 1


def test_no_store_tiles_are_not_cached(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200, content=png(), headers={"Content-Type": "image/png", "Cache-Control": "no-store"}
        )

    provider = PDOK(httpx.Client(transport=httpx.MockTransport(handle)))
    tiles = MapTiles(tmp_path, provider)
    for _ in range(2):
        assert tiles.get("street", 0, 0, 0)[2] == 0
    assert len(requests) == 2
    assert not list((tmp_path / "map-cache").glob("*.tile"))


@pytest.mark.parametrize(
    "kind,z,x,y", [("evil", 0, 0, 0), ("street", 21, 0, 0), ("street", 1, 3, 0), ("aerial", 2, 0, -1)]
)
def test_invalid_tiles_do_not_call_network(tmp_path, kind, z, x, y):
    def handle(request):
        raise AssertionError("Must reject invalid coordinates before networking")

    tiles = MapTiles(tmp_path, PDOK(httpx.Client(transport=httpx.MockTransport(handle))))
    with pytest.raises(UserError):
        tiles.get(kind, z, x, y)
