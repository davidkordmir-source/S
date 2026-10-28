"""Create a ZIP64 portable package and bounded parts with a small offline installer."""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PART_BYTES = 1500 * 1024**2


def main():
    root = Path("dist")
    shutil.copy2("AI-MODELS.json", root / "AI-MODELS.json")
    archive = root / "AtlasScout-Offline-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as output:
        for path in sorted((root / "AtlasScout").rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(root))
    records, complete = [], hashlib.sha256()
    with archive.open("rb") as source:
        index = 0
        while True:
            first = source.read(4 * 1024**2)
            if not first:
                break
            index += 1
            name = f"AtlasScout-Windows.part{index:02d}"
            digest, count = hashlib.sha256(), 0
            with (root / name).open("wb") as target:
                chunk = first
                while chunk:
                    target.write(chunk)
                    digest.update(chunk)
                    complete.update(chunk)
                    count += len(chunk)
                    if count >= PART_BYTES:
                        break
                    chunk = source.read(min(4 * 1024**2, PART_BYTES - count))
            records.append(dict(name=name, size=count, sha256=digest.hexdigest()))
    manifest = dict(
        version="0.8.0",
        archive_sha256=complete.hexdigest(),
        archive_bytes=archive.stat().st_size,
        parts=records,
    )
    metadata = root / "AtlasScout-Package.json"
    metadata.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive.unlink()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            "AtlasScout-Setup",
            "--add-data",
            f"{metadata.resolve()}:package",
            "--paths",
            "scripts",
            "scripts/offline-installer.py",
        ],
        check=True,
    )
    installer = root / "AtlasScout-Setup.exe"
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    lines = [f"{digest}  {installer.name}"] + [f"{p['sha256']}  {p['name']}" for p in records]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
