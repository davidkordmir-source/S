import numpy as np
import pytest
from conftest import TestEngine, png

from atlas.errors import UserError
from atlas.store import Store, normalize


def test_search_persists_normalizes_and_ranks(tmp_path):
    store = Store(tmp_path)
    for index, red in enumerate((False, True)):
        path = store.images / f"{index}.jpg"
        path.write_bytes(png("red" if red else "blue"))
        store.put(id=str(index), title=str(index), lat=52.3, lon=4.8, source="fixture", image=path)
        store.embed(str(index), TestEngine.vector(red) * 4)
    # Verify a fresh store can retrieve previously persisted embeddings.
    reopened = Store(tmp_path)
    result = reopened.search(TestEngine.vector(True) * 3)
    assert [row["id"] for row in result] == ["1", "0"]
    assert [row["score"] for row in result] == [1.0, 0.0]
    assert reopened.counts() == {"total": 2, "indexed": 2}


def test_replacing_image_invalidates_index(tmp_path):
    store = Store(tmp_path)
    path = store.images / "building.jpg"
    arguments = dict(id="a", title="A", lat=52, lon=4, source="fixture", image=path)
    store.put(**arguments)
    store.embed("a", TestEngine.vector(True))
    store.put(**arguments)
    assert store.counts() == {"total": 1, "indexed": 0}
    with pytest.raises(UserError):
        store.search(TestEngine.vector(True))


@pytest.mark.parametrize("vector", [np.zeros(512), np.ones(511), np.full(512, np.nan), np.ones((1, 512))])
def test_invalid_embeddings_rejected(vector):
    with pytest.raises(ValueError):
        normalize(vector)


def test_image_paths_cannot_escape_store(tmp_path):
    store = Store(tmp_path / "store")
    outside = tmp_path / "private.jpg"
    outside.write_bytes(png())
    store.put(id="bad", title="bad", lat=0, lon=0, source="fixture", image=outside)
    assert store.image_path("bad") is None


def test_search_is_restricted_to_selected_geographic_area(tmp_path):
    store = Store(tmp_path)
    for id, lat, lon in (("amsterdam", 52.3728, 4.8936), ("tehran", 35.7, 51.4)):
        store.put(id=id, title=id, lat=lat, lon=lon, source="fixture", image=store.images / f"{id}.jpg")
        store.embed(id, TestEngine.vector(True))
    results = store.search(TestEngine.vector(True), area={"lat": 35.7, "lon": 51.4, "radius": 250})
    assert [row["id"] for row in results] == ["tehran"]
    with pytest.raises(UserError):
        store.search(TestEngine.vector(True), area={"lat": 0, "lon": 0, "radius": 250})
