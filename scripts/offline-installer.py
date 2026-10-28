"""Offline Windows installer: verify and read archive parts without joining on disk."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
import zipfile
from pathlib import Path


class JoinedReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.length = sum(self.sizes)
        self.position = 0
        self.streams = [p.open("rb") for p in self.paths]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.length + offset
            if whence == 2
            else -1
        )
        if value < 0:
            raise ValueError("Invalid seek")
        self.position = value
        return value

    def read(self, size=-1):
        remaining = (
            max(0, self.length - self.position)
            if size < 0
            else min(size, max(0, self.length - self.position))
        )
        chunks, start = [], 0
        for stream, length in zip(self.streams, self.sizes, strict=True):
            if self.position < start + length and remaining:
                stream.seek(self.position - start)
                chunk = stream.read(min(remaining, start + length - self.position))
                chunks.append(chunk)
                self.position += len(chunk)
                remaining -= len(chunk)
            start += length
        return b"".join(chunks)

    def close(self):
        for stream in self.streams:
            stream.close()
        super().close()


def verify(manifest, folder, progress=lambda value, text: None):
    paths, complete, done = [], hashlib.sha256(), 0
    for item in manifest["parts"]:
        path = folder / item["name"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError("Missing or incomplete file: " + item["name"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                digest.update(chunk)
                complete.update(chunk)
                done += len(chunk)
                progress(done / manifest["archive_bytes"] * 35, "Verifying " + item["name"])
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Checksum mismatch: " + item["name"])
        paths.append(path)
    if complete.hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Complete archive checksum mismatch")
    return paths


def install(manifest, folder, destination, progress=lambda value, text: None):
    if destination.exists():
        raise ValueError("Choose a NEW destination folder. Existing installations are never overwritten.")
    paths = verify(manifest, folder, progress)
    staging = destination.with_name(destination.name + ".installing")
    if staging.exists():
        raise ValueError("A previous installation is unfinished. Choose another destination folder.")
    try:
        with JoinedReader(paths) as source, zipfile.ZipFile(source) as archive:
            total = sum(i.file_size for i in archive.infolist())
            if shutil.disk_usage(destination.parent).free < total + 256 * 1024**2:
                raise ValueError("Insufficient disk space")
            staging.mkdir()
            done = 0
            for item in archive.infolist():
                relative = Path(item.filename)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "AtlasScout":
                    raise ValueError("Invalid archive path")
                archive.extract(item, staging)
                done += item.file_size
                progress(35 + done / max(total, 1) * 65, "Extracting " + relative.name)
        (staging / "AtlasScout").rename(destination)
        staging.rmdir()
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return destination / "AtlasScout.exe"


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    folder = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path("dist").resolve()
    data = (
        Path(sys._MEIPASS) / "package/AtlasScout-Package.json"
        if getattr(sys, "frozen", False)
        else folder / "AtlasScout-Package.json"
    )
    manifest = json.loads(data.read_text(encoding="utf-8"))
    if "--verify-package" in sys.argv:
        paths = verify(manifest, folder)
        with JoinedReader(paths) as source, zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC validation failed")
            assert "AtlasScout/AtlasScout.exe" in archive.namelist()
        window = tk.Tk()
        window.withdraw()
        window.update()
        window.destroy()
        (folder / "installer-smoke.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    parts=len(paths),
                    archive_sha256=manifest["archive_sha256"],
                    archive_bytes=manifest["archive_bytes"],
                    all_zip_crc_checked=True,
                    native_tk_window=True,
                )
            ),
            encoding="utf-8",
        )
        return
    window = tk.Tk()
    window.title("Atlas Scout 0.8 — Offline Setup")
    window.geometry("610x280")
    ttk.Label(window, text="Atlas Scout — Offline AI", font=("Segoe UI", 18)).pack(pady=18)
    ttk.Label(
        window,
        text="Keep this installer and ALL AtlasScout-Windows.part files in the same folder.\nNo internet, Python or Node.js is needed. Choose a NEW installation folder.",
        justify="center",
    ).pack()
    selected = tk.StringVar(
        value=str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AtlasScout-0.8.0")
    )
    ttk.Entry(window, textvariable=selected, width=75).pack(pady=12)
    status = tk.StringVar(value="Ready. Your saved training data is kept separately.")
    ttk.Label(window, textvariable=status, wraplength=570).pack()
    bar = ttk.Progressbar(window, maximum=100, length=560)
    bar.pack(pady=8)
    actions = ttk.Frame(window)
    actions.pack()

    def choose():
        parent = filedialog.askdirectory(title="Choose parent folder")
        if parent:
            selected.set(str(Path(parent) / "AtlasScout-0.8.0"))

    def begin():
        target = Path(selected.get()).expanduser().resolve()
        if not target.parent.is_dir():
            messagebox.showerror("Atlas Scout", "Choose an existing parent folder.")
            return
        start.config(state="disabled")
        browse.config(state="disabled")

        def work():
            try:
                executable = install(
                    manifest, folder, target, lambda value, text: window.after(0, update, value, text)
                )
                window.after(0, finished, executable)
            except Exception as exc:
                window.after(0, failed, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def update(value, text):
        bar["value"] = value
        status.set(text)

    def failed(text):
        messagebox.showerror("Installation failed", text)
        start.config(state="normal")
        browse.config(state="normal")

    def finished(executable):
        status.set("Installed successfully: " + str(executable))
        if messagebox.askyesno("Atlas Scout", "Installation verified. Open Atlas Scout now?"):
            subprocess.Popen([str(executable)], cwd=executable.parent)
        window.destroy()

    browse = ttk.Button(actions, text="Choose folder", command=choose)
    browse.pack(side="left", padx=6)
    start = ttk.Button(actions, text="Verify and install", command=begin)
    start.pack(side="left", padx=6)
    window.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        report = (
            Path(sys.executable).parent / "installer-error.log"
            if getattr(sys, "frozen", False)
            else Path("installer-error.log")
        )
        report.write_text(traceback.format_exc(), encoding="utf-8")
        if "--verify-package" in sys.argv:
            raise SystemExit(1) from None
        raise
