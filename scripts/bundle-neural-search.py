"""Prepare verified real image-guided detection and vision-language model data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

DETECTOR = "google/owlv2-base-patch16-ensemble"
ASSISTANT = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
BASE_ASSISTANT = "Qwen/Qwen3-VL-4B-Instruct"


def checksum(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            value.update(chunk)
    return value.hexdigest()


def verified_file(repo, revision, filename, cache):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    meta = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
    source = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache))
    if meta.commit_hash != revision or meta.size != source.stat().st_size:
        raise RuntimeError("Official model metadata mismatch: " + filename)
    if re.fullmatch("[a-f0-9]{64}", meta.etag or ""):
        actual = checksum(source)
    elif re.fullmatch("[a-f0-9]{40}", meta.etag or ""):
        value = hashlib.sha1(f"blob {source.stat().st_size}\0".encode())
        with source.open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                value.update(chunk)
        actual = value.hexdigest()
    else:
        raise RuntimeError("Unsupported official checksum: " + filename)
    if actual != meta.etag:
        raise RuntimeError("Official model checksum mismatch: " + filename)
    return source, checksum(source)


def prepare(cache, output, lock):
    from huggingface_hub import HfApi

    api = HfApi()
    records = {}
    # Prefer the model authors' quantization; only the official runtime org is
    # an alternate source of this SAME 4B model, never a smaller substitute.
    from huggingface_hub.errors import RepositoryNotFoundError

    assistant_repository = lock.get("assistant-model", {}).get("repository", ASSISTANT)
    if assistant_repository not in (ASSISTANT, "ggml-org/Qwen3-VL-4B-Instruct-GGUF"):
        raise RuntimeError("Unexpected assistant source in the model lock")
    if not lock.get("assistant-model", {}).get("revision"):
        try:
            api.model_info(assistant_repository)
        except RepositoryNotFoundError:
            assistant_repository = "ggml-org/Qwen3-VL-4B-Instruct-GGUF"
            api.model_info(assistant_repository)
    for kind, repository in [("detector", DETECTOR), ("assistant-model", assistant_repository)]:
        revision = lock.get(kind, {}).get("revision")
        info = api.model_info(repository, revision=revision, files_metadata=True)
        if revision and info.sha != revision:
            raise RuntimeError("Model revision did not resolve to the source lock")
        revision = info.sha
        print("Preparing real neural search model:", kind, repository, revision, flush=True)
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        names = [f.rfilename for f in info.siblings]
        hashes = {}
        if kind == "detector":
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            card = info.card_data.to_dict() if info.card_data else {}
            if card.get("license") != "apache-2.0":
                raise RuntimeError("Unexpected OWLv2 model license; review before distribution")
            wanted = {
                "config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "README.md",
                "LICENSE",
            }
            weights = [n for n in names if n.endswith(".safetensors") and "/" not in n]
            if not weights:
                weights = ["pytorch_model.bin"]
            wanted.update(weights)
            wanted.update(n for n in names if n.endswith(".safetensors.index.json"))
            for filename in names:
                if filename not in wanted:
                    continue
                source, digest = verified_file(repository, revision, filename, cache)
                shutil.copy2(source, folder / filename)
                hashes[filename] = digest
            model = Owlv2ForObjectDetection.from_pretrained(
                str(folder),
                local_files_only=True,
                use_safetensors=bool([n for n in weights if n.endswith(".safetensors")]),
                dtype=torch.float32,
                attn_implementation="sdpa",
            ).eval()
            processor = Owlv2Processor.from_pretrained(str(folder), local_files_only=True, backend="pil")
            model.half().save_pretrained(folder, safe_serialization=True)
            processor.save_pretrained(folder)
            (folder / "pytorch_model.bin").unlink(missing_ok=True)
            record = dict(
                repository=repository,
                revision=revision,
                source_files=hashes,
                license="Apache-2.0",
                conversion="fp16 storage; local float32 inference; native image-guided and text-guided OWLv2 heads",
            )
            del model, processor
        else:
            language = [
                n
                for n in names
                if n.lower().endswith(".gguf")
                and "q4_k_m" in n.lower()
                and "mmproj" not in n.lower()
                and "/" not in n
            ]
            projector = [
                n
                for n in names
                if n.lower().endswith(".gguf")
                and "mmproj" in n.lower()
                and re.search(r"(?:^|[-_.])(?:f16|fp16)(?:[-_.]|$)", n.lower())
                and "/" not in n
            ]
            if len(language) != 1 or len(projector) != 1:
                raise RuntimeError(
                    "Expected one Q4_K_M language model and one F16 multimodal projector: "
                    + json.dumps(dict(language=language, projector=projector))
                )
            original = api.model_info(BASE_ASSISTANT)
            card = original.card_data.to_dict() if original.card_data else {}
            if card.get("license") != "apache-2.0":
                raise RuntimeError("Unexpected Qwen model license; review before distribution")
            for filename in language + projector + [n for n in names if n in ("README.md", "LICENSE")]:
                source, digest = verified_file(repository, revision, filename, cache)
                shutil.copy2(source, folder / filename)
                hashes[filename] = digest
            license_path, license_digest = verified_file(BASE_ASSISTANT, original.sha, "LICENSE", cache)
            shutil.copy2(license_path, folder / "LICENSE")
            for previous in folder.glob("*.gguf"):
                if previous.name not in language + projector:
                    previous.unlink()
            record = dict(
                repository=repository,
                revision=revision,
                source_files=hashes,
                original_repository=BASE_ASSISTANT,
                original_revision=original.sha,
                original_license_sha256=license_digest,
                license="Apache-2.0",
                language_file=language[0],
                projector_file=projector[0],
                conversion="Published GGUF Q4_K_M decoder and F16 vision projector; loaded by pinned local llama.cpp",
            )
        (folder / "atlas-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["files"] = {p.name: checksum(p) for p in folder.iterdir() if p.is_file()}
        records[kind] = record
        print(
            "::notice title=Neural model source::"
            + json.dumps(dict(component=kind, repository=repository, revision=revision)),
            flush=True,
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".model-cache")
    parser.add_argument("--output", default=".model-bundle")
    parser.add_argument("--lock", default="packaging/neural-models.json")
    args = parser.parse_args()
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(name, None)
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    records = prepare(str(cache), output, lock)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"].update(records)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output.parent / "AI-MODELS.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Verified real object detector and offline vision-language model are ready.", flush=True)


if __name__ == "__main__":
    main()
