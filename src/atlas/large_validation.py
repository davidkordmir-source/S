"""Reproducible 64-bit disk-address and bounded-preview validation.

The fixture is sparse: it tests a real tile beyond 1 TiB, not a physical TiB
of imagery or performance on every satellite codec.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import warnings
from pathlib import Path

import numpy as np

TIB = 1 << 40
FAR_X, FAR_Y = 28672, 22016


def mark_sparse(file):
    if os.name == "nt":
        import ctypes
        import msvcrt

        returned = ctypes.c_ulong()
        control = ctypes.WinDLL("kernel32", use_last_error=True).DeviceIoControl
        control.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_void_p,
        ]
        control.restype = ctypes.c_int
        if not control(
            msvcrt.get_osfhandle(file.fileno()), 0x900C4, None, 0, None, 0, ctypes.byref(returned), None
        ):
            raise OSError(ctypes.get_last_error(), "Cannot mark the test fixture as sparse")


def extend_sparse(file, size):
    # CPython 3.12 uses Windows CRT _chsize_s for truncate(), which explicitly
    # writes zeroes when extending. A seek + one-byte write preserves the hole.
    mark_sparse(file)
    if size > os.fstat(file.fileno()).st_size:
        file.seek(size - 1)
        file.write(b"\0")


def allocated_size(path):
    if os.name != "nt":
        return path.stat().st_blocks * 512
    import ctypes
    from ctypes import wintypes

    high = wintypes.DWORD()
    function = ctypes.WinDLL("kernel32", use_last_error=True).GetCompressedFileSizeW
    function.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    function.restype = wintypes.DWORD
    ctypes.set_last_error(0)
    low = function(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise OSError(ctypes.get_last_error(), "Cannot measure sparse file allocation")
    return (high.value << 32) | low


def make_sparse_scene(path: Path, far_offset: int = TIB + 4096):
    import rasterio
    from rasterio.windows import Window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=30000,
            height=24000,
            count=3,
            dtype="uint8",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
            SPARSE_OK="YES",
        ) as dataset:
            blue = np.zeros((3, 512, 512), dtype=np.uint8)
            blue[2] = 230
            dataset.write(blue, window=Window(0, 0, 512, 512))
            red = np.zeros_like(blue)
            red[0] = 230
            dataset.write(red, window=Window(FAR_X, FAR_Y, 512, 512))
    # Relocate a genuine encoded tile and its LONG8 offset, preserving TIFF data.
    with path.open("r+b") as file:
        assert file.read(4) == b"II+\0"
        file.seek(8)
        directory = struct.unpack("<Q", file.read(8))[0]
        file.seek(directory)
        count = struct.unpack("<Q", file.read(8))[0]
        entries = {}
        for _ in range(count):
            tag, kind, length, value = struct.unpack("<HHQQ", file.read(20))
            entries[tag] = kind, length, value
        assert entries[324][0] == 16  # 64-bit tile offsets
        index = (FAR_Y // 512) * ((30000 + 511) // 512) + FAR_X // 512
        pointer = entries[324][2] + index * 8
        file.seek(pointer)
        old = struct.unpack("<Q", file.read(8))[0]
        payload_size = 512 * 512 * 3  # uncompressed, pixel-interleaved RGB
        file.seek(old)
        payload = file.read(payload_size)
        assert len(payload) == payload_size
        mark_sparse(file)
        file.seek(far_offset)
        file.write(payload)
        file.seek(pointer)
        file.write(struct.pack("<Q", far_offset))
    return far_offset


def peak_memory():
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    value = Counters()
    value.cb = ctypes.sizeof(value)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not get_info(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("Cannot measure process memory")
    return value.PeakWorkingSetSize


def validate_large_files(root: Path):
    from PIL import Image

    from .jobs import Job
    from .scene_source import SceneReader
    from .workshop import Workshop

    baseline = peak_memory()
    with tempfile.TemporaryDirectory(prefix="atlas-large-", dir=root) as temporary:
        folder = Path(temporary)
        source = folder / "one-tib-bigtiff.tif"
        offset = make_sparse_scene(source)
        with SceneReader(source) as reader:
            assert (reader.width, reader.height) == (30000, 24000)
            crop = reader.read((FAR_X, FAR_Y, FAR_X + 128, FAR_Y + 128))
            assert crop.getpixel((64, 64)) == (230, 0, 0)
            overview = reader.read(maximum=1200)
            assert max(overview.size) <= 1200
        # Model-free validation of source, coordinates, preview, crops and exports.
        workshop = Workshop(folder / "workspace", None)
        scene = workshop.import_scene_file(source, source.name, Job("import"))["scene"]
        assert scene["storage"] == "linked" and scene["source_size"] > TIB
        tile = workshop.tile(scene["id"], 0, FAR_X // 512, FAR_Y // 512)
        with Image.open(tile) as picture:
            assert picture.getpixel((64, 64))[0] > 220
        concept = workshop.new_concept("large-raster")["id"]
        workshop.add_box(concept, scene["id"], dict(x=FAR_X, y=FAR_Y, width=128, height=128), 1)
        example = workshop.examples(concept)[0]
        assert example["bbox"]["x"] == FAR_X
        picture = workshop.annotated(
            {"scene": scene["id"], "detections": [{"index": 0, "bbox": example["bbox"]}]}
        )
        assert max(picture.size) == 4096
        assert Workshop(folder / "workspace", None).scene_path(scene["id"]) == source
        logical_size = source.stat().st_size
        allocated = allocated_size(source)
        assert allocated < 10 * 1024 * 1024, "Test fixture must remain physically sparse"
    delta = max(0, peak_memory() - baseline)
    assert delta < 600 * 1024 * 1024, f"Large-file smoke exceeded memory budget: {delta}"
    report = dict(
        passed=True,
        logical_file_bytes=logical_size,
        real_tile_offset=offset,
        original_pixels=[30000, 24000],
        tile_rgb=[230, 0, 0],
        peak_memory_growth_bytes=delta,
        allocated_bytes=allocated,
        scope="Sparse BigTIFF with a real tile beyond 1 TiB; not a physical 1 TiB dataset or universal codec benchmark",
    )
    (root / "large-files-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
