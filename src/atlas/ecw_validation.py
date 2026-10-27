"""Regression ECWs whose directories exceed the former RAM admission limit."""

from __future__ import annotations

import struct
import time
from pathlib import Path

from .large_validation import extend_sparse, peak_memory

SIDE = 300_000
BLOCK = 128
COUNT = ((SIDE + BLOCK - 1) // BLOCK) ** 2


def make_large_directory(path: Path, version: int):
    record = struct.pack(">BIII", 0, SIDE, SIDE, 1)
    if version == 2:
        header = bytearray(89)
        header[:6] = bytes([101, 2, 1, 1, 1, 4])
        header[6:22] = struct.pack(">IIHHHH", SIDE * 2, SIDE * 2, 1, 1, BLOCK, BLOCK)
        # Compressed directory: long runs of zero coefficient words.
        n = (COUNT + 1) * 4
        runs, rest = divmod(n, 32767)
        encoded = bytes([6]) + b"\xff\xff" * runs + (struct.pack("<H", 0x8000 | rest) if rest else b"")
        path.write_bytes(header + record + struct.pack(">I", len(encoded)) + encoded)
        with path.open("r+b") as file:
            extend_sparse(file, COUNT + 2)
    else:

        def box(kind, payload):
            return struct.pack(">I4s", len(payload) + 8, kind) + payload

        image = bytearray(22)
        image[2] = 1
        image[3:13] = struct.pack(">IIH", SIDE * 2, SIDE * 2, 1)
        image[13:19] = struct.pack(">BBHH", 1, 4, BLOCK, BLOCK)
        image[20:22] = struct.pack(">H", 1)
        prefix = box(b"ecw ", b"\r\n\x87\n\0\0\0\0") + box(b"ecwh", box(b"ihdr", image))
        stream_size = len(record) + 5 + COUNT * 12
        with path.open("wb") as file:
            file.write(prefix)
            file.write(struct.pack(">I4s", stream_size + 8, b"ecwc"))
            file.write(record)
            file.write(struct.pack(">IB", COUNT * 12 + 1, 1))
            extend_sparse(file, len(prefix) + 8 + stream_size)
    return COUNT


def validate_ecw_directories(root: Path):
    import json
    import tempfile

    from .scene_source import SceneReader

    root.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="atlas-ecw-index-", dir=root) as temporary:
        folder = Path(temporary)
        for version in (2, 3):
            path = folder / f"large-v{version}.ecw"
            count = make_large_directory(path, version)
            before = peak_memory()
            began = time.monotonic()
            for _ in range(2):
                with SceneReader(path, cache_directory=folder / "index") as reader:
                    image = reader.read((399000, 399000, 399128, 399128))
                    assert image.size == (128, 128) and image.getextrema() == ((0, 0),) * 3
            delta = max(0, peak_memory() - before)
            assert delta < 128 * 1024 * 1024, f"ECW directory unexpectedly entered RAM: {delta}"
            report.append(
                dict(
                    version=version,
                    blocks=count,
                    former_admission_bytes=count * (64 if version == 2 else 32),
                    peak_memory_growth=delta,
                    seconds=round(time.monotonic() - began, 3),
                    cache_bytes=sum(p.stat().st_size for p in (folder / "index").glob("*.idx"))
                    if (folder / "index").exists()
                    else 0,
                )
            )
    result = {
        "status": "passed",
        "cases": report,
        "note": "Synthetic valid empty tiles with compressed v2 and raw v3 directories; real ECW v2/v3 imagery is tested separately.",
    }
    (root / "ecw-index-smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
