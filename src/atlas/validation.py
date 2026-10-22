from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from .engine import Engine
from .imagery import PDOK
from .jobs import Job
from .store import Store


def validate_packaged_ai(root: Path) -> dict:
    """Exercise real dependencies and models in the frozen Windows package."""
    engine = Engine(root)
    logging.info("Packaged AI check: preparing pinned models")
    engine.prepare()
    with tempfile.TemporaryDirectory(prefix="atlas-validation-", dir=root) as directory:
        store = Store(Path(directory))
        provider = PDOK()
        try:
            logging.info("Packaged AI check: retrieving five actual Amsterdam building images")
            provider.prepare(store, Job("validation"), 52.3728, 4.8936, 250, 5)
            for building in store.pending():
                store.embed(building["id"], engine.image(Path(building["image"])))
            reference = store.list(1)[0]
            matches = store.search(engine.image(store.image_path(reference["id"])))
            assert matches[0]["id"] == reference["id"] and matches[0]["score"] > 0.99
            persian = engine.text("ساختمانی با سقف قرمز")
            english = engine.text("a building with a red roof")
            alignment = float(np.dot(persian, english))
            assert alignment > 0.5
            assert store.search(persian) and store.search(english)
            result = {
                "status": "passed",
                "library": store.counts(),
                "same_image_score": matches[0]["score"],
                "persian_english_alignment": round(alignment, 5),
            }
            (root / "ai-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            logging.info("Packaged AI check passed: %s", result)
            return result
        finally:
            provider.close()
