"""Validate real CLIP inference against an already prepared local library.

Use --offline to require cached models. This is a retrieval sanity check, not a
measurement of search accuracy on a labeled building-identification dataset.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from atlas.config import data_directory
from atlas.engine import Engine
from atlas.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    root = data_directory()
    store = Store(root)
    if not store.counts()["indexed"]:
        raise SystemExit("Prepare and index a real building library before running this validation.")
    engine = Engine(root)
    engine.prepare()
    reference = store.list(1)[0]
    image = store.image_path(reference["id"])
    if image is None:
        raise SystemExit("The reference image is missing.")
    matches = store.search(engine.image(image))
    assert matches[0]["score"] >= 0.99, "Same-image retrieval should have near-unit similarity"
    own = next((item for item in matches if item["id"] == reference["id"]), None)
    assert own and own["score"] >= matches[0]["score"] - 0.0001, (
        "Reference must rank at the top, allowing exact duplicates"
    )
    persian = engine.text("ساختمانی با سقف قرمز")
    english = engine.text("a building with a red roof")
    alignment = float(np.dot(persian, english))
    assert np.isfinite(alignment) and alignment > 0.5, "Equivalent text prompts should align"
    first, second = store.search(persian), store.search(english)
    assert first and second
    print(
        json.dumps(
            {
                "mode": "cached-models-offline" if args.offline else "online-model-loading-allowed",
                "library": store.counts(),
                "same_image_score": own["score"],
                "reference_top_rank": matches[0]["id"] == reference["id"],
                "persian_english_embedding_similarity": round(alignment, 5),
                "persian_results": len(first),
                "english_results": len(second),
                "same_top_text_result": first[0]["id"] == second[0]["id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
