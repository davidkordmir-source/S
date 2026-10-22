"""Run with Python + Playwright installed, after npm run build in ui/.

The server uses deterministic test embeddings. This verifies UI wiring and
failure handling, NOT model quality, real model inference, or PDOK access.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def png(color):
    file = io.BytesIO()
    Image.new("RGB", (160, 160), color).save(file, format="PNG")
    return file.getvalue()


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
                    raise RuntimeError("UI test server failed to start") from exc
                time.sleep(0.1)
        else:
            raise RuntimeError("UI test server did not become ready")
        with sync_playwright() as playwright:
            chromium = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
            browser = playwright.chromium.launch(
                headless=True, **({"executable_path": chromium} if chromium else {})
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:8766/?token=ui-test-only-session")
            expect(page.get_by_role("heading", name="از یک نشانه، به یک مکان")).to_be_visible()
            expect(page.get_by_role("button", name="پیدا کردن ساختمان‌ها")).to_be_disabled()
            page.get_by_role("button", name="ساختمان با سقف قرمز و حیاط بزرگ").click()
            expect(page.get_by_role("textbox")).to_have_value("ساختمان با سقف قرمز و حیاط بزرگ")
            page.get_by_role("tab", name="با تصویر").click()
            image_input = page.locator('input[type=file][accept*="image/jpeg"]')
            image_input.set_input_files(
                {"name": "reference.png", "mimeType": "image/png", "buffer": png("red")}
            )
            expect(page.get_by_alt_text("تصویر مرجع انتخاب‌شده")).to_be_visible()
            page.get_by_role("button", name="حذف تصویر").click()
            page.get_by_role("tab", name="با توضیح").click()
            page.get_by_role("button", name="نقشهٔ خیابان", exact=True).click()
            page.get_by_role("button", name="بزرگ‌نمایی", exact=True).click()
            page.get_by_role("button", name="تصویر هوایی", exact=True).click()
            page.get_by_role("button", name="کوچک‌نمایی", exact=True).click()

            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as file:
                file.writestr(
                    "buildings.json",
                    json.dumps(
                        {
                            "buildings": [
                                {
                                    "id": "red",
                                    "title": "Red roof fixture",
                                    "lat": 52.3728,
                                    "lon": 4.8936,
                                    "image": "red.png",
                                },
                                {
                                    "id": "blue",
                                    "title": "Blue roof fixture",
                                    "lat": 52.373,
                                    "lon": 4.894,
                                    "image": "blue.png",
                                },
                            ]
                        }
                    ),
                )
                file.writestr("red.png", png("red"))
                file.writestr("blue.png", png("blue"))
            page.locator('input[type=file][accept*="zip"]').set_input_files(
                {"name": "test.zip", "mimeType": "application/zip", "buffer": payload.getvalue()}
            )
            expect(page.get_by_role("status").filter(has_text="تصاویر وارد شدند")).to_be_visible(
                timeout=15000
            )
            page.get_by_role("button", name="ساخت نمایه", exact=True).click()
            expect(page.get_by_role("status").filter(has_text="نمایه آماده است")).to_be_visible(timeout=15000)
            page.get_by_role("button", name="پیدا کردن ساختمان‌ها").click()
            expect(page.locator(".result-card")).to_have_count(2, timeout=15000)
            expect(page.locator(".result-card").first).to_contain_text("Red roof fixture")
            page.locator(".result-card").first.click()
            expect(page.locator(".selected-building")).to_be_visible()
            with page.expect_download() as download:
                page.get_by_role("button", name="خروجی GeoJSON").click()
            output = json.loads(Path(download.value.path()).read_text())
            assert len(output["features"]) == 2

            page.get_by_role("tab", name="با تصویر").click()
            image_input.set_input_files({"name": "blue.png", "mimeType": "image/png", "buffer": png("blue")})
            page.get_by_role("button", name="پیدا کردن ساختمان‌ها").click()
            expect(page.locator(".result-card").first).to_contain_text("Blue roof fixture", timeout=15000)
            page.locator(".navigation").get_by_role("button", name="کتابخانه").click()
            expect(page.locator(".result-card")).to_have_count(2)
            page.get_by_role("button", name="تنظیمات", exact=True).click()
            expect(page.get_by_role("dialog")).to_be_visible()
            page.keyboard.press("Escape")
            expect(page.get_by_role("dialog")).to_have_count(0)
            page.get_by_role("button", name="راهنما", exact=True).click()
            expect(page.get_by_role("heading", name="از کجا شروع کنم؟")).to_be_visible()
            page.get_by_role("button", name="شروع کاوش", exact=True).click()
            page.get_by_role("button", name="دریافت حداکثر ۳۰ تصویر از محدوده").click()
            expect(page.get_by_role("alert")).to_contain_text("403", timeout=15000)
            for width, height in ((1440, 1050), (1080, 800)):
                page.set_viewport_size({"width": width, "height": height})
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            assert not errors, errors
            browser.close()
        print(
            "UI smoke passed: import, index, Persian text query, image query, ranking, map selection, GeoJSON export, dialogs, network failure, and window sizes."
        )
        print("Uses test embeddings; real AI and live imagery are separate checks.")
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
