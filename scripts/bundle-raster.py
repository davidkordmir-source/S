"""Prepare a self-contained Windows GDAL + ECW reader from verified OSGeo4W packages.

This runs only at build time. No installers or package post-install scripts run.
Original third-party license notices are included alongside the native tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

SITE = "https://download.osgeo.org/osgeo4w/v2/"


def download(url: str) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "AtlasScout-build/0.2.1"}), timeout=120
    ) as response:
        return response.read()


def packages_from_index(text: str) -> dict:
    packages = {}
    for block in re.split(r"^@ ", text, flags=re.MULTILINE)[1:]:
        lines = block.splitlines()
        entry = {}
        for line in lines[1:]:
            if line.startswith("["):
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                entry[key] = value
        packages[lines[0].strip()] = entry
    return packages


def verified_file(entry: str, cache: Path) -> tuple[Path, str]:
    fields = shlex.split(entry)
    if len(fields) != 3:
        raise RuntimeError("OSGeo4W file metadata has an unexpected structure")
    relative, size, expected = fields
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise RuntimeError("Invalid package path in the official index")
    algorithms = {32: "md5", 64: "sha256", 128: "sha512"}
    algorithm = algorithms.get(len(expected))
    if not algorithm or not re.fullmatch("[a-fA-F0-9]+", expected):
        raise RuntimeError("Unknown checksum type in the official package index")
    target = cache / path.name
    if not target.exists():
        target.write_bytes(download(SITE + relative))
    data = target.read_bytes()
    if len(data) != int(size) or hashlib.new(algorithm, data).hexdigest() != expected.lower():
        raise RuntimeError(f"Official package verification failed: {path.name}")
    return target, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".raster-bundle")
    parser.add_argument("--cache", default=".raster-cache")
    args = parser.parse_args()
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    index_data = download(SITE + "x86_64/setup.ini")
    packages = packages_from_index(index_data.decode("utf-8-sig"))
    selected = set()

    def require(name):
        if name in selected:
            return
        if name not in packages:
            raise RuntimeError(f"OSGeo4W dependency not found: {name}")
        selected.add(name)
        for dependency in packages[name].get("requires", "").split():
            require(dependency)

    for name in ("gdal", "gdal-ecw"):
        require(name)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    notices = output / "third-party-licenses"
    notices.mkdir()
    manifest = dict(source=SITE, index_sha256=hashlib.sha256(index_data).hexdigest(), packages={})
    for name in sorted(selected):
        entry = packages[name]
        print("Preparing native raster dependency:", name, entry.get("version"), flush=True)
        if "install" not in entry:
            raise RuntimeError(f"Package has no install archive: {name}")
        archive, digest = verified_file(entry["install"], cache)
        with tarfile.open(archive) as file:
            file.extractall(output, filter="data")
        record = dict(version=entry.get("version"), archive=archive.name, sha256=digest)
        if "license" in entry:
            fields = shlex.split(entry["license"])
            if len(fields) == 3:
                license_file, license_digest = verified_file(entry["license"], cache)
                shutil.copyfile(license_file, notices / f"{name}.txt")
                record["license_sha256"] = license_digest
            else:
                (notices / f"{name}.txt").write_text(entry["license"], encoding="utf-8")
        elif name == "gdal-ecw":
            raise RuntimeError("The ECW redistribution license must accompany the reader")
        manifest["packages"][name] = record
    # Keep the Microsoft redistributable runtime alongside the native helper,
    # instead of relying on a system-wide VC++ installation on the user's PC.
    runtime = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "concrt140.dll",
    ):
        if (runtime / name).is_file():
            shutil.copyfile(runtime / name, output / "bin" / name)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        PATH=str(output / "bin") + os.pathsep + environment.get("PATH", ""),
        GDAL_DRIVER_PATH=str(output / "apps/gdal/lib/gdalplugins"),
        GDAL_DATA=str(output / "apps/gdal/share/gdal"),
        PROJ_DATA=str(output / "share/proj"),
        PROJ_NETWORK="OFF",
        GDAL_PAM_ENABLED="NO",
    )
    command = output / "bin/gdalinfo.exe"
    result = subprocess.run(
        [str(command), "--formats"], env=environment, capture_output=True, text=True, check=True
    )
    if not re.search(r"^\s*ECW\s", result.stdout, re.MULTILINE):
        raise RuntimeError("The actual GDAL binary does not expose the ECW reader")
    reports = []
    for name in ("jrc.ecw", "stefan_full_rgba_ecwv3_meta.ecw"):
        source = Path("tests/data/ecw") / name
        result = subprocess.run(
            [str(command), "-json", str(source.resolve())],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        assert data["driverShortName"] == "ECW" and min(data["size"]) > 0
        target = cache / (name + ".png")
        subprocess.run(
            [
                str(output / "bin/gdal_translate.exe"),
                "-of",
                "PNG",
                "-ot",
                "Byte",
                "-b",
                "1",
                str(source.resolve()),
                str(target),
            ],
            env=environment,
            check=True,
        )
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        reports.append(
            dict(file=name, driver=data["driverShortName"], size=data["size"], bands=len(data["bands"]))
        )
    print("ECW v2/v3 native decode passed:", json.dumps(reports), flush=True)
    (output / "native-validation.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
