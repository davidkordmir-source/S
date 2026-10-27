"""Exercise the shipped single-screenshot workflow through the browser UI."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atlas.similarity_validation import make_similarity_scene  # noqa: E402


def main():
    executable = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    process = subprocess.Popen([str(executable), "tests/ui_server.py"], cwd=ROOT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8766/", timeout=1).close()
                break
            except urllib.error.URLError:
                if process.poll() is not None:
                    raise RuntimeError("UI server exited") from None
                time.sleep(0.1)
        picture, sample, truth = make_similarity_scene()
        stream = io.BytesIO()
        picture.save(stream, format="PNG")
        with sync_playwright() as playwright:
            chromium = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
            browser = playwright.chromium.launch(
                headless=True, **({"executable_path": chromium} if chromium else {})
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            external = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda r: (
                    external.append(r.url)
                    if r.url.startswith(("http:", "https:"))
                    and not r.url.startswith("http://127.0.0.1:8766/")
                    else None
                ),
            )
            page.goto("http://127.0.0.1:8766/?token=ui-test-only-session")
            expect(page.get_by_role("tab", name="مشابه این تصویر", exact=True)).to_have_attribute(
                "aria-selected", "true"
            )
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {"name": "scene.png", "mimeType": "image/png", "buffer": stream.getvalue()}
            )
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=15000)
            page.get_by_test_id("reference-input").set_input_files(
                {"name": "screenshot.jpg", "mimeType": "image/jpeg", "buffer": sample}
            )
            expect(page.locator(".reference-preview img")).to_be_visible()
            expect(page.get_by_role("spinbutton", name="عرض نمونه روی تصویر")).to_have_value("64")
            slider = page.get_by_role("slider", name="حداقل شباهت", exact=True)
            slider.focus()
            slider.press("Home")
            for _ in range(38):
                slider.press("ArrowUp")
            expect(slider).to_have_value("0.78")
            expect(page.get_by_role("button", name="پیداکردن موارد مشابه", exact=True)).to_be_enabled(
                timeout=10000
            )
            page.get_by_role("button", name="پیداکردن موارد مشابه", exact=True).click()
            expect(page.locator(".result-card")).to_have_count(len(truth), timeout=60000)
            expect(page.locator(".detection-box")).to_have_count(len(truth), timeout=10000)
            expect(page.locator(".map-level-legend")).to_be_visible()
            assert "٪" in page.locator(".detection-box text").first.text_content()
            assert "٪ شباهت" in page.locator(".result-photo b").first.text_content()
            expect(page.locator(".level-filters button[aria-pressed=true]")).to_have_count(1)
            page.locator(".level-filters .level-review").click()
            expect(page.locator(".level-filters .level-review")).to_have_attribute("aria-pressed", "true")
            expect(page.locator(".result-card")).to_have_count(0)
            expect(page.locator(".detection-box")).to_have_count(0)
            page.locator(".level-filters button").first.click()
            expect(page.locator(".result-card")).to_have_count(len(truth))
            assert (
                page.get_by_role("combobox", name="هدف آموزشی").count() == 0
            )  # no concept or negative examples
            page.locator(".result-photo").last.click()
            expect(page.locator(".detection-box.selected")).to_have_count(1)
            expect(page.get_by_role("img", name="نمونهٔ اصلی برای مقایسه")).to_be_visible()
            expect(page.get_by_role("img", name="نتیجهٔ انتخاب‌شده برای مقایسه")).to_be_visible()
            expect(page.locator(".evidence-bars")).to_be_visible()
            page.wait_for_timeout(400)
            page.screenshot(path="/tmp/atlas-050-ranked-results.png", full_page=True)
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="CSV", exact=True).click()
            csv = Path(downloaded.value.path()).read_text(encoding="utf-8-sig")
            assert len(csv.strip().splitlines()) == len(truth) + 1
            assert "similarity_percent,level" in csv
            page.get_by_role("button", name="نمایش کل تصویر", exact=True).click()
            page.get_by_role("button", name="انتخاب نمونه از روی تصویر", exact=True).click()
            bounds = page.locator(".scene-full").bounding_box()
            sx, sy = bounds["width"] / picture.width, bounds["height"] / picture.height
            page.mouse.move(bounds["x"] + 100 * sx, bounds["y"] + 100 * sy)
            page.mouse.down()
            page.mouse.move(bounds["x"] + 164 * sx, bounds["y"] + 148 * sy, steps=6)
            page.mouse.up()
            expect(page.locator(".reference-box")).to_have_count(1)
            expect(page.get_by_role("status").filter(has_text="نمونهٔ مرجع انتخاب شد")).to_be_visible()
            # A genuine File in the paste event follows the same clipboard path.
            page.evaluate("""async () => {
                const response=await fetch(document.querySelector('.reference-preview img').src);
                const data=new DataTransfer();data.items.add(new File([await response.blob()],'pasted.png',{type:'image/png'}));
                window.dispatchEvent(new ClipboardEvent('paste',{clipboardData:data,bubbles:true}));
            }""")
            expect(page.locator(".teaching-gallery article")).to_have_count(2)  # duplicate pasted pixels do not add another view
            expect(page.locator(".reference-preview b")).to_have_text("screenshot.jpg")
            for width, height in [(1080, 800), (1440, 1050)]:
                page.set_viewport_size({"width": width, "height": height})
                assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
            page.wait_for_timeout(400)
            expect(page.locator(".similar-footer")).to_be_in_viewport()
            page.screenshot(path="/tmp/atlas-050-similarity-ui.png", full_page=True)
            assert not errors, errors
            assert not external, external
            browser.close()
        print(
            "Screenshot UI passed: file reference, one-shot AI matching, all expected patterns and overlays, result focus, CSV, reference selection from scene, clipboard paste, two window sizes, zero external requests."
        )
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
