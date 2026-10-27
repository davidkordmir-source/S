"""Build-time network operation. Runtime model loading is always offline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from atlas.config import DETAIL_MODEL, DETAIL_REVISION, IMAGE_MODEL, IMAGE_REVISION, TEXT_MODEL, TEXT_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".model-cache")
    parser.add_argument("--output", default=".model-bundle")
    parser.add_argument(
        "--detail-only", action="store_true", help="Add the detail model to an existing bundle"
    )
    args = parser.parse_args()
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download
    from sentence_transformers import SentenceTransformer

    manifest = {"format": "atlas-model-bundle-v1", "models": {}}
    if args.detail_only:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    models = [
        ("image", IMAGE_MODEL, IMAGE_REVISION),
        ("text", TEXT_MODEL, TEXT_REVISION),
        ("detail", DETAIL_MODEL, DETAIL_REVISION),
    ]
    for kind, repository, revision in models:
        if args.detail_only and kind != "detail":
            continue
        print(f"Preparing pinned {kind} model: {repository} @ {revision}", flush=True)
        model = None
        if kind == "detail":
            snapshot_download(
                repository,
                revision=revision,
                cache_dir=str(cache),
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
            )
        else:
            model = SentenceTransformer(
                repository, revision=revision, cache_folder=str(cache), device="cpu", trust_remote_code=False
            )
        snapshot = cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
        if not (snapshot / ("config.json" if kind == "detail" else "modules.json")).is_file():
            raise RuntimeError("Pinned snapshot did not resolve to the expected location")
        hashes = {}
        destination = output / kind
        if destination.exists():
            shutil.rmtree(destination)
        # Follow verified hub-cache links so the portable Windows package does
        # not depend on symlink privileges or the original cache directory.
        for original in snapshot.rglob("*"):
            if not original.is_file():
                continue
            target = destination / original.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with original.open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    digest.update(chunk)
            relative = original.relative_to(snapshot).as_posix()
            metadata = get_hf_file_metadata(hf_hub_url(repository, relative, revision=revision))
            expected = metadata.etag
            if metadata.commit_hash != revision or metadata.size != original.stat().st_size:
                raise RuntimeError(f"Model metadata mismatch: {relative}")
            # Hub cache names can be Xet identifiers rather than content SHA256.
            # Verify against the pinned repository's official ETag instead.
            if re.fullmatch("[0-9a-f]{64}", expected or ""):
                actual = digest.hexdigest()
            elif re.fullmatch("[0-9a-f]{40}", expected or ""):
                git_hash = hashlib.sha1(f"blob {original.stat().st_size}\0".encode())
                git_hash.update(original.read_bytes())
                actual = git_hash.hexdigest()
            else:
                raise RuntimeError(f"Unsupported model checksum: {relative}")
            if actual != expected:
                raise RuntimeError(f"Model checksum failed: {original.name}")
            shutil.copyfile(original, target)
            hashes[str(original.relative_to(snapshot)).replace("\\", "/")] = digest.hexdigest()
        manifest["models"][kind] = {"repository": repository, "revision": revision, "files": hashes}
        del model
        gc.collect()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Offline model bundle prepared and verified:", output, flush=True)


if __name__ == "__main__":
    main()
