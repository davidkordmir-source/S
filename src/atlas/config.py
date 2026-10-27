from __future__ import annotations

import os
from pathlib import Path

IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_REVISION = "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
TEXT_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"
DETAIL_MODEL = "facebook/dinov2-small"
DETAIL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
# These models share a 512-dimensional CLIP embedding space. Changing either
# model requires a new index version and re-indexing all images.
INDEX_VERSION = f"clip-b32-multilingual-v1:{IMAGE_REVISION[:12]}:{TEXT_REVISION[:12]}:1"
MODEL_CACHE = "models"
WFS_URL = "https://service.pdok.nl/lv/bag/wfs/v2_0"
WMS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
DEFAULT_CENTER = {"lat": 52.3728, "lon": 4.8936}


def data_directory() -> Path:
    configured = os.environ.get("ATLAS_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "AtlasScout"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "atlas-scout"
