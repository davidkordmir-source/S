"""Exercise the offline annotation/training/search UI with deterministic test embeddings."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from offline_support import scene_bytes  # noqa: E402


def main():
    executable = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    process = subprocess.Popen([str(executable), "tests/ui_server.py"], cwd=ROOT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8766/", timeout=1).close()
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                if process.poll() is not None:
                    raise RuntimeError("UI server failed to start") from exc
                time.sleep(0.1)
        else:
            raise RuntimeError("UI server did not become ready")
        with sync_playwright() as playwright:
            chromium = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
            browser = playwright.chromium.launch(
                headless=True, **({"executable_path": chromium} if chromium else {})
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors, external = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    external.append(request.url)
                    if request.url.startswith(("http:", "https:"))
                    and not request.url.startswith("http://127.0.0.1:8766/")
                    else None
                ),
            )
            page.goto("http://127.0.0.1:8766/?token=ui-test-only-session")
            page.get_by_role("tab", name="دسته‌بند نمونه‌ای", exact=True).click()
            expect(page.get_by_role("heading", level=1)).to_contain_text("یاد بده")
            expect(page.get_by_role("button", name="جست‌وجو روی تصویر", exact=True)).to_be_disabled()
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {"name": "user-satellite.png", "mimeType": "image/png", "buffer": scene_bytes()}
            )
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=15000)
            page.get_by_role("textbox", name="نام هدف جدید").fill("سقف قرمز")
            page.get_by_role("button", name="ساخت هدف", exact=True).click()
            expect(page.get_by_role("combobox", name="هدف آموزشی")).to_be_visible()
            for index, (x, y, label) in enumerate([(16, 16, 1), (144, 16, 1), (16, 112, 0), (144, 112, 0)]):
                page.locator(".drawing-tools").get_by_role(
                    "button", name="نمونه درست" if label else "نمونه غلط", exact=True
                ).click()
                bounds = page.locator(".scene-full").bounding_box()
                sx, sy = bounds["width"] / 256, bounds["height"] / 192
                page.mouse.move(bounds["x"] + x * sx, bounds["y"] + y * sy)
                page.mouse.down()
                page.mouse.move(bounds["x"] + (x + 64) * sx, bounds["y"] + (y + 64) * sy, steps=5)
                page.mouse.up()
                expect(page.locator(".sample-card")).to_have_count(index + 1, timeout=10000)
            page.get_by_role("button", name="اندازه از نمونه‌ها", exact=True).click()
            page.get_by_role("button", name="آموزش با نمونه‌های من", exact=True).click()
            expect(page.get_by_role("button", name="آموزش آماده است · آموزش مجدد", exact=True)).to_be_enabled(
                timeout=15000
            )
            page.get_by_role("button", name="جست‌وجو روی تصویر", exact=True).click()
            expect(page.locator(".result-card").first).to_be_visible(timeout=15000)
            page.locator(".result-photo").first.click()
            expect(page.locator(".detection-box.selected")).to_have_count(1)
            with page.expect_download() as download:
                page.get_by_role("button", name="CSV", exact=True).click()
            csv = Path(download.value.path()).read_text(encoding="utf-8-sig")
            assert "x,y,width,height,model_score" in csv and len(csv.strip().splitlines()) > 1
            with page.expect_download() as download:
                page.get_by_role("button", name="پیش‌نمایش نشانه‌گذاری‌شده", exact=True).click()
            assert Image.open(download.value.path()).size == (256, 192)
            page.locator(".feedback-actions").first.get_by_role("button", name="غلط", exact=True).click()
            expect(page.get_by_role("button", name="به‌روزرسانی آموزش", exact=True)).to_be_enabled()
            expect(page.get_by_role("button", name="جست‌وجو روی تصویر", exact=True)).to_be_disabled()
            page.get_by_role("button", name="به‌روزرسانی آموزش", exact=True).click()
            expect(page.get_by_role("button", name="آموزش آماده است · آموزش مجدد", exact=True)).to_be_enabled(
                timeout=15000
            )
            page.locator(".collection-tabs").get_by_role("button", name="نمونه‌های آموزشی").click()
            expect(page.locator(".sample-card")).to_have_count(5)
            with page.expect_download() as download:
                page.get_by_role("button", name="ذخیرهٔ آموزش", exact=True).click()
            pack = Path(download.value.path()).read_bytes()
            with zipfile.ZipFile(io.BytesIO(pack)) as archive:
                assert "training.json" in archive.namelist()
            page.locator('input[accept=".zip"]').set_input_files(
                {"name": "training.zip", "mimeType": "application/zip", "buffer": pack}
            )
            expect(page.get_by_role("status").filter(has_text="نمونه‌ها وارد شدند")).to_be_visible(
                timeout=15000
            )
            expect(page.get_by_role("combobox", name="هدف آموزشی").locator("option")).to_have_count(2)
            page.get_by_role("tab", name="با توضیح", exact=True).click()
            page.get_by_role("textbox", name="چه چیزی را پیدا کند؟").fill("ساختمان با سقف قرمز")
            page.get_by_role("button", name="جست‌وجو روی تصویر", exact=True).click()
            expect(page.locator(".result-card").first).to_be_visible(timeout=15000)
            page.get_by_role("button", name="نمایش کل تصویر", exact=True).click()
            page.get_by_role("button", name="محدوده جست‌وجو", exact=True).click()
            bounds = page.locator(".scene-full").bounding_box()
            page.mouse.move(bounds["x"] + 10, bounds["y"] + 10)
            page.mouse.down()
            page.mouse.move(
                bounds["x"] + bounds["width"] * 0.6, bounds["y"] + bounds["height"] * 0.6, steps=5
            )
            page.mouse.up()
            expect(page.get_by_role("button", name="حذف محدوده", exact=True)).to_be_visible()
            page.get_by_role("button", name="حذف محدوده", exact=True).click()
            page.get_by_role("button", name="تنظیمات", exact=True).click()
            expect(page.get_by_role("dialog")).to_contain_text("تمام مراحل آفلاین")
            page.keyboard.press("Escape")
            page.get_by_role("button", name="راهنما", exact=True).click()
            expect(page.get_by_role("heading", name="از نمونه تا نتیجه", exact=True)).to_be_visible()
            page.get_by_text("فرمت‌های قابل خواندن", exact=True).click()
            expect(page.locator(".format-grid")).to_contain_text("ECW")
            page.get_by_role("button", name="شروع می‌کنم", exact=True).click()
            accepted = page.locator('input[type=file]:not([multiple])[accept^=".png"]').get_attribute(
                "accept"
            )
            assert all(
                extension in accepted.split(",") for extension in (".ecw", ".heic", ".jp2", ".avif", ".psd")
            )
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {
                    "name": "satellite.heic",
                    "mimeType": "image/heic",
                    "buffer": (ROOT / "tests/data/formats/basic.heic").read_bytes(),
                }
            )
            expect(page.get_by_role("combobox", name="تصویر فعال")).to_have_value(
                __import__("re").compile(".+")
            )
            expect(page.locator(".scene-full.loaded")).to_have_attribute(
                "alt", "satellite.heic", timeout=15000
            )
            page.get_by_role("tab", name="با نمونه‌های من", exact=True).click()
            page.locator(".collection-tabs").get_by_role("button", name="نمونه‌های آموزشی").click()
            expect(page.get_by_role("button", name="فایل نمونهٔ درست", exact=True)).to_be_enabled()
            page.locator("input[type=file][multiple]").set_input_files(
                {
                    "name": "sample.jp2",
                    "mimeType": "image/jp2",
                    "buffer": (ROOT / "tests/data/formats/basic.jp2").read_bytes(),
                }
            )
            try:
                expect(page.locator(".sample-card")).to_have_count(6, timeout=10000)
            except AssertionError:
                print("Image upload notice:", page.locator(".notice").all_text_contents())
                print(
                    "Image upload state:",
                    page.evaluate(
                        "fetch('/api/state').then(r=>r.json()).then(s=>s.concepts.map(c=>({id:c.id,positive:c.positive,negative:c.negative})))"
                    ),
                )
                raise
            for width, height in [(1440, 1050), (1080, 800)]:
                page.set_viewport_size({"width": width, "height": height})
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.screenshot(path="/tmp/atlas-offline-ui.png", full_page=True)
            assert not errors, errors
            assert not external, external
            browser.close()
        print(
            "Offline UI passed: image import, four drawn training boxes, training, scan, focus, CSV/PNG exports, feedback, retraining, training ZIP roundtrip, Persian text search, ROI, help/settings, two window sizes, zero external requests."
        )
        print(
            "UI uses test embeddings. The separate real-model offline validation checks actual CPU inference."
        )
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
