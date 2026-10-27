"""Verify and package aerial/segmentation weights and a DirectML-compatible DINO graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


def verified_file(repo, revision, filename, cache):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
    path = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache))
    sha = hashlib.sha256()
    git_sha = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            sha.update(chunk)
            git_sha.update(chunk)
    digest = sha.hexdigest()
    actual = digest if re.fullmatch("[a-f0-9]{64}", metadata.etag or "") else git_sha.hexdigest()
    if metadata.commit_hash != revision or path.stat().st_size != metadata.size or actual != metadata.etag:
        raise RuntimeError("Pinned model checksum/metadata mismatch: " + filename)
    return path, digest


def remoteclip(cache, output):
    import torch
    from safetensors.torch import save_file
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    repo = "chendelong/RemoteCLIP"
    rev = "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    path, digest = verified_file(repo, rev, "RemoteCLIP-ViT-B-32.pt", cache)
    source = torch.load(path, map_location="cpu", weights_only=True)
    config = CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=32,
        projection_dim=512,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
    )
    model = CLIPVisionModelWithProjection(config).eval()
    state = {}
    mapping = {
        "conv1.weight": "embeddings.patch_embedding.weight",
        "class_embedding": "embeddings.class_embedding",
        "positional_embedding": "embeddings.position_embedding.weight",
        "ln_pre.weight": "pre_layrnorm.weight",
        "ln_pre.bias": "pre_layrnorm.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
    }
    for a, b in mapping.items():
        state["vision_model." + b] = source["visual." + a]
    state["visual_projection.weight"] = source["visual.proj"].T.contiguous()
    for i in range(12):
        a = f"visual.transformer.resblocks.{i}."
        b = f"vision_model.encoder.layers.{i}."
        for suffix in ["weight", "bias"]:
            for name, tensor in zip(
                ("q_proj", "k_proj", "v_proj"),
                source[a + "attn.in_proj_" + suffix].chunk(3, dim=0),
                strict=True,
            ):
                state[b + "self_attn." + name + "." + suffix] = tensor
        for old, new in [
            ("attn.out_proj", "self_attn.out_proj"),
            ("ln_1", "layer_norm1"),
            ("ln_2", "layer_norm2"),
            ("mlp.c_fc", "mlp.fc1"),
            ("mlp.c_proj", "mlp.fc2"),
        ]:
            for suffix in ["weight", "bias"]:
                state[b + new + "." + suffix] = source[a + old + "." + suffix]
    model.load_state_dict(state, strict=True)
    folder = output / "aerial"
    folder.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(folder)
    save_file(
        {k: v.half().contiguous() for k, v in state.items()},
        str(folder / "model.safetensors"),
        metadata={"format": "pt"},
    )
    # Verify the layout conversion against an independent, original-weight ViT forward.
    torch.set_num_threads(4)
    sample = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    with torch.inference_mode():
        x = (
            torch.nn.functional.conv2d(sample, source["visual.conv1.weight"].float(), stride=32)
            .flatten(2)
            .transpose(1, 2)
        )
        x = (
            torch.cat([source["visual.class_embedding"].float()[None, None].expand(1, 1, -1), x], dim=1)
            + source["visual.positional_embedding"].float()
        )
        x = torch.nn.functional.layer_norm(
            x, (768,), source["visual.ln_pre.weight"].float(), source["visual.ln_pre.bias"].float(), 1e-5
        )
        for i in range(12):
            p = f"visual.transformer.resblocks.{i}."
            z = torch.nn.functional.layer_norm(
                x, (768,), source[p + "ln_1.weight"].float(), source[p + "ln_1.bias"].float(), 1e-5
            )
            q, k, v = torch.nn.functional.linear(
                z, source[p + "attn.in_proj_weight"].float(), source[p + "attn.in_proj_bias"].float()
            ).chunk(3, dim=-1)
            attn = (
                torch.nn.functional.scaled_dot_product_attention(
                    *[t.reshape(1, 50, 12, 64).transpose(1, 2) for t in (q, k, v)]
                )
                .transpose(1, 2)
                .reshape(1, 50, 768)
            )
            x = x + torch.nn.functional.linear(
                attn, source[p + "attn.out_proj.weight"].float(), source[p + "attn.out_proj.bias"].float()
            )
            z = torch.nn.functional.layer_norm(
                x, (768,), source[p + "ln_2.weight"].float(), source[p + "ln_2.bias"].float(), 1e-5
            )
            z = torch.nn.functional.gelu(
                torch.nn.functional.linear(
                    z, source[p + "mlp.c_fc.weight"].float(), source[p + "mlp.c_fc.bias"].float()
                )
            )
            x = x + torch.nn.functional.linear(
                z, source[p + "mlp.c_proj.weight"].float(), source[p + "mlp.c_proj.bias"].float()
            )
        expected = (
            torch.nn.functional.layer_norm(
                x[:, 0],
                (768,),
                source["visual.ln_post.weight"].float(),
                source["visual.ln_post.bias"].float(),
                1e-5,
            )
            @ source["visual.proj"].float()
        )
        actual = model(pixel_values=sample).image_embeds
        if not torch.allclose(expected, actual, rtol=3e-4, atol=3e-4):
            raise RuntimeError("RemoteCLIP conversion parity failed")
        from safetensors.torch import load_file

        model.load_state_dict(load_file(folder / "model.safetensors"), strict=True)
        reloaded = model(pixel_values=sample).image_embeds
        stored_cosine = float(torch.nn.functional.cosine_similarity(expected, reloaded).min())
        if stored_cosine < 0.9999:
            raise RuntimeError("Stored RemoteCLIP conversion parity failed")
    return dict(
        repository=repo,
        revision=rev,
        source_sha256=digest,
        conversion="Vision only, Transformers layout, fp16 storage; GELU as official OpenCLIP ViT-B-32 defaults",
        parity_max_error=float((expected - actual).abs().max()),
        stored_fp16_cosine=stored_cosine,
    )


def segment(cache, output):
    from safetensors.torch import load_file, save_file
    from transformers import Sam2Config, Sam2Model

    repo = "facebook/sam2.1-hiera-tiny"
    rev = "de431c4043854a71d8101e17995dfe596bf101a5"
    folder = output / "segment"
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in ["config.json", "model.safetensors", "README.md", "preprocessor_config.json"]:
        path, digest = verified_file(repo, rev, filename, cache)
        hashes[filename] = digest
        (folder / filename).write_bytes(path.read_bytes())
    original = json.loads((folder / "config.json").read_text())
    config = Sam2Config(
        vision_config=original["vision_config"],
        prompt_encoder_config=original["prompt_encoder_config"],
        mask_decoder_config=original["mask_decoder_config"],
    )
    config.architectures = ["Sam2Model"]
    model = Sam2Model(config)
    expected = model.state_dict()
    source = load_file(folder / "model.safetensors")
    state = {k: source[k] for k in expected}
    model.load_state_dict(state, strict=True)
    save_file(
        {k: v.half().contiguous() for k, v in state.items()},
        str(folder / "model.safetensors"),
        metadata={"format": "pt"},
    )
    config.save_pretrained(folder)
    return dict(
        repository=repo,
        revision=rev,
        source_files=hashes,
        conversion="Image segmentation subset; tracking-only weights removed, fp16 storage, runtime float32; strict image-model key/shape validation",
    )


def onnx_detail(output):
    import onnxruntime as ort
    import torch
    from transformers import Dinov2Model

    ort.disable_telemetry_events()

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Dinov2Model.from_pretrained(
                str(output / "detail"), local_files_only=True, attn_implementation="eager"
            ).eval()

        def forward(self, pixels):
            return self.model(pixel_values=pixels).last_hidden_state

    torch.set_num_threads(4)
    model = Graph().eval()
    folder = output / "detail-onnx"
    folder.mkdir(exist_ok=True)
    sample = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (sample,),
        str(folder / "model.onnx"),
        input_names=["pixels"],
        output_names=["tokens"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={"pixels": {0: "batch", 2: "height", 3: "width"}, "tokens": {0: "batch", 1: "patches"}},
    )
    session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
    errors = []
    for h, w, b in [(224, 224, 1), (168, 224, 2), (448, 448, 1)]:
        x = torch.linspace(-1, 1, b * 3 * h * w).reshape(b, 3, h, w)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {"pixels": x.numpy()})[0]
        if not np.allclose(expected, actual, atol=5e-4, rtol=5e-4):
            raise RuntimeError("ONNX feature parity failed")
        errors.append(float(np.max(np.abs(expected - actual))))
    return dict(source="pinned local DINOv2", opset=18, dynamic_shapes=True, cpu_parity_max_error=max(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".model-cache")
    parser.add_argument("--output", default=".model-bundle")
    parser.add_argument("--only", choices=["aerial", "segment", "onnx"])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    cache = str(Path(args.cache).resolve())
    os.environ["HF_HOME"] = cache
    os.environ["HF_XET_CACHE"] = str(Path(cache) / "xet")
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        os.environ.pop(key, None)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, function in [
        ("aerial", lambda: remoteclip(cache, output)),
        ("segment", lambda: segment(cache, output)),
        ("onnx", lambda: onnx_detail(output)),
    ]:
        if args.only and name != args.only:
            continue
        print("Preparing verified enhanced component:", name, flush=True)
        metadata = function()
        folder = output / ("detail-onnx" if name == "onnx" else name)
        metadata["files"] = {
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }
        from filelock import FileLock

        with FileLock(str(manifest_path) + ".lock"):
            manifest = json.loads(manifest_path.read_text())
            manifest["models"][name] = metadata
            manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Enhanced component ready:", name, flush=True)


if __name__ == "__main__":
    main()
