"""Build the pinned local multimodal server; runtime always uses local model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REVISION = "a7af149a2a7768d4b1f6f02c6ce26a17fba11376"
REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"


def run(command):
    subprocess.run([str(p) for p in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".assistant-cache")
    parser.add_argument("--output", default=".assistant-bundle")
    args = parser.parse_args()
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    source, build = cache / "source", cache / "build"
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").is_dir():
        run(["git", "init", source])
        run(["git", "-C", source, "remote", "add", "origin", REPOSITORY])
    run(["git", "-C", source, "fetch", "--depth=1", "origin", REVISION])
    run(["git", "-C", source, "checkout", "--detach", REVISION])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("Multimodal runtime source pin mismatch")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError(
            "The pinned runtime source has local changes; refusing to distribute an unverified build"
        )
    cmake = Path(sys.executable).parent / ("cmake.exe" if os.name == "nt" else "cmake")
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    server = next(build.rglob(executable), None) if build.exists() else None
    cache_file = build / "CMakeCache.txt"
    cache_text = cache_file.read_text(encoding="utf-8", errors="replace") if cache_file.is_file() else ""
    requirements = {
        "LLAMA_BUILD_TESTS": "OFF",
        "LLAMA_BUILD_EXAMPLES": "OFF",
        "LLAMA_BUILD_SERVER": "ON",
        "LLAMA_BUILD_UI": "OFF",
        "LLAMA_CONNECT": "OFF",
        "LLAMA_OPENSSL": "OFF",
        "LLAMA_SUBPROCESS": "OFF",
        "GGML_NATIVE": "OFF",
        "GGML_BACKEND_DL": "ON",
        "GGML_CPU_ALL_VARIANTS": "ON",
        "GGML_CUDA": "OFF",
        "GGML_VULKAN": "OFF",
    }
    reused = False
    if server is not None and all(f"{key}:BOOL={value}" in cache_text for key, value in requirements.items()):
        # Reuse only our runnable build of this exact pinned source and flags.
        # The actual frozen model tests still run on every distributed build.
        version = subprocess.run([str(server), "--version"], capture_output=True, text=True)
        reused = version.returncode == 0 and REVISION[:7] in version.stdout + version.stderr
    if reused:
        print("Reusing the verified pinned native runtime build", flush=True)
    else:
        run(
            [
                cmake,
                "-S",
                source,
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_APP=OFF",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DLLAMA_BUILD_UI=OFF",
                "-DLLAMA_USE_PREBUILT_UI=OFF",
                "-DLLAMA_CONNECT=OFF",
                "-DLLAMA_OPENSSL=OFF",
                "-DLLAMA_SUBPROCESS=OFF",
                "-DGGML_NATIVE=OFF",
                "-DGGML_BACKEND_DL=ON",
                "-DGGML_CPU_ALL_VARIANTS=ON",
                "-DGGML_CUDA=OFF",
                "-DGGML_VULKAN=OFF",
            ]
        )
        run([cmake, "--build", build, "--config", "Release", "--target", "llama-server", "--parallel", "2"])
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    server = next(build.rglob(executable), None)
    if server is None:
        raise RuntimeError("The local multimodal server was not built")
    output.mkdir(parents=True, exist_ok=True)
    for path in server.parent.iterdir():
        if path.name == executable or path.suffix.lower() in (".dll", ".so", ".dylib") or ".so." in path.name:
            shutil.copy2(path, output / path.name)
    shutil.copy2(source / "LICENSE", output / "LICENSE")
    (output / "NOTICE.md").write_text(
        f"llama.cpp local multimodal runtime\nSource: {REPOSITORY}\nRevision: {REVISION}\nMIT License included. Built without HTTPS, subprocess tools or a web UI. Automatic model access is disabled at launch. Atlas supplies local model paths and --offline at launch.\n",
        encoding="utf-8",
    )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(dict(repository=REPOSITORY, revision=REVISION, files=hashes), indent=2), encoding="utf-8"
    )
    result = subprocess.run([str(output / executable), "--help"], capture_output=True, text=True, check=True)
    if "--offline" not in result.stdout + result.stderr:
        raise RuntimeError("The bundled runtime lacks offline-mode support")
    print("Verified local multimodal runtime ready:", output, flush=True)


if __name__ == "__main__":
    main()
