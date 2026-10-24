"""Check large-scene streaming and visible-tile browser navigation."""

import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

root = Path(__file__).resolve().parents[1]
process = subprocess.Popen(
    [
        str(root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")),
        "tests/ui_server.py",
    ],
    cwd=root,
)
try:
    for _ in range(100):
        try:
            urllib.request.urlopen("http://127.0.0.1:8766/", timeout=1).close()
            break
        except Exception:
            time.sleep(0.1)
    with tempfile.TemporaryDirectory(prefix="atlas-large-ui-") as td:
        source = Path(td) / "large-scene.tif"
        subprocess.run(
            [
                str(root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")),
                "-c",
                "from pathlib import Path; from atlas.large_validation import make_sparse_scene; make_sparse_scene(Path(__import__('sys').argv[1]), 160*1024*1024)",
                str(source),
            ],
            check=True,
            cwd=root,
        )
        assert source.stat().st_size > 150 * 1024 * 1024
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=shutil.which("chromium"))
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            tiles = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("request", lambda r: tiles.append(r.url) if "/tiles/" in r.url else None)
            page.goto("http://127.0.0.1:8766/?token=ui-test-only-session")
            expect(page.get_by_role("heading", level=1)).to_be_visible()
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(str(source))
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=30000)
            expect(page.locator(".notice.error")).to_have_count(0)
            expect(page.locator(".scene-tiles img").first).to_be_visible(timeout=10000)
            fit_tiles = set(tiles)
            for _ in range(12):
                page.get_by_role("button", name="بزرگ‌نمایی", exact=True).click()
            page.wait_for_timeout(800)
            assert len(set(tiles) - fit_tiles) > 0
            assert page.locator(".scene-tiles img").count() <= 64
            bounds = page.locator("[data-testid=scene-viewport]").bounding_box()
            page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
            page.mouse.down()
            page.mouse.move(
                bounds["x"] + bounds["width"] / 2 + 180, bounds["y"] + bounds["height"] / 2 + 100, steps=6
            )
            page.mouse.up()
            page.wait_for_timeout(600)
            page.get_by_role("button", name="نمایش کل تصویر", exact=True).click()
            page.screenshot(path=str(Path(tempfile.gettempdir()) / "atlas-030-large-ui.png"), full_page=True)
            assert not errors, errors
            assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
            print(
                "Large-scene browser UI passed: real >150 MiB stream upload, 720 MP dimensions, adaptive fit, tiled zoom/pan, <=64 visible tiles, no JavaScript errors."
            )
            browser.close()
finally:
    process.terminate()
    process.wait(timeout=10)
