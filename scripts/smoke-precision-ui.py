"""Browser wiring including GPU-render fallback; actual model checks live in precision_validation."""

from __future__ import annotations

import io
import os
import re
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
from atlas.teaching_validation import make_teaching_scene  # noqa:E402


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
        scene, examples, _, _ = make_teaching_scene()
        stream = io.BytesIO()
        scene.save(stream, "PNG")
        with sync_playwright() as playwright:
            chromium = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
            browser = playwright.chromium.launch(
                headless=True, **({"executable_path": chromium} if chromium else {})
            )
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            errors, external = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
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
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {"name": "scene.png", "mimeType": "image/png", "buffer": stream.getvalue()}
            )
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=15000)
            page.wait_for_function(
                "['webgl','fallback'].includes(document.querySelector('.scene-gpu')?.dataset.backend)"
            )
            renderer = page.locator(".scene-gpu").get_attribute("data-backend")
            if renderer == "webgl":
                page.wait_for_function("Number(document.querySelector('.scene-gpu')?.dataset.loadedTiles)>0")
            page.get_by_test_id("reference-input").set_input_files(
                {"name": "positive.png", "mimeType": "image/png", "buffer": examples[0]}
            )
            expect(page.locator(".teaching-gallery article")).to_have_count(1)
            group = page.get_by_role("combobox", name="هدف تصویری", exact=True).input_value()
            page.get_by_title("جداکردن هدف در positive.png", exact=True).click()
            editor = page.get_by_role("dialog", name="جداکردن خود هدف")
            expect(editor).to_be_visible()
            surface = editor.locator(".mask-surface")
            surface.click(position={"x": 40, "y": 40})
            editor.get_by_role("button", name="جداکردن با هوش مصنوعی", exact=True).click()
            expect(editor.locator(".mask-preview")).to_be_visible(timeout=15000)
            page.screenshot(path="/tmp/atlas-070-mask-ui.png")
            editor.get_by_role("button", name="اعمال همین ماسک", exact=True).click()
            expect(editor).to_have_count(0)
            expect(page.locator(".teaching-gallery article")).to_have_count(1)
            members = page.request.get(
                f"http://127.0.0.1:8766/api/workspace/reference-groups/{group}"
            ).json()["references"]
            assert members[0]["has_mask"]
            page.get_by_title("جداکردن هدف در " + members[0]["name"], exact=True).click()
            editor.get_by_role("button", name="بازگشت به عکس کامل", exact=True).click()
            expect(editor).to_have_count(0)
            page.get_by_role("combobox", name="نوع عکس آموزشی", exact=True).select_option("0")
            page.get_by_test_id("reference-input").set_input_files(
                {"name": "negative.png", "mimeType": "image/png", "buffer": examples[1]}
            )
            expect(page.locator(".teaching-gallery article")).to_have_count(2)
            page.locator(".precision-tools summary").click()
            tools = page.locator(".precision-tools")
            tools.get_by_role("button", name="آموزش اختصاصی و ارزیابی", exact=True).click()
            expect(
                tools.get_by_role("button", name="آموزش اختصاصی آماده است · آموزش مجدد", exact=True)
            ).to_be_enabled(timeout=15000)
            assert not page.locator(".evaluation-report").count()
            tools.get_by_role("button", name="ساخت نمایهٔ آفلاین", exact=True).click()
            expect(tools.get_by_role("button", name="ادامهٔ نمایه‌سازی", exact=True)).to_be_disabled(
                timeout=15000
            )
            tools.get_by_role("combobox", name="حالت جست‌وجوی صحنه", exact=True).select_option("fast")
            expect(tools.locator(".precision-warning")).to_be_visible()
            search = page.locator(".similar-footer .primary-button")
            expect(search).to_be_enabled(timeout=10000)
            search.click()
            expect(page.locator(".result-card").first).to_be_visible(timeout=60000)
            expect(page.locator(".similarity-summary .precision-warning")).to_be_visible()
            run = re.search(
                r"/runs/([^/]+)/", page.locator(".result-photo img").first.get_attribute("src")
            ).group(1)
            page.locator(".result-card").first.get_by_role("button", name="غلط", exact=True).click()
            expect(
                page.locator(".result-card").first.get_by_role(
                    "button", name="به غیرهدف‌ها اضافه شد", exact=True
                )
            ).to_be_enabled(timeout=15000)
            csv = page.request.get(f"http://127.0.0.1:8766/api/workspace/runs/{run}/csv")
            assert (
                csv.ok
                and "aerial_similarity" in csv.text()
                and "geometry_status" in csv.text()
                and "fast" in csv.text()
            )
            page.get_by_role("button", name="تنظیمات", exact=True).click()
            hardware = page.locator(".hardware-panel")
            expect(hardware).to_be_visible()
            hardware.get_by_role("combobox", name="پردازش سخت افزاری", exact=True).select_option("cpu")
            expect(hardware.get_by_role("combobox", name="پردازش سخت افزاری", exact=True)).to_be_enabled()
            hardware.get_by_role("combobox", name="حالت کارایی سیستم", exact=True).select_option("balanced")
            expect(hardware.get_by_role("combobox", name="حالت کارایی سیستم", exact=True)).to_be_enabled()
            expect(hardware.get_by_role("combobox", name="حالت کارایی سیستم", exact=True)).to_have_value(
                "balanced"
            )
            page.screenshot(path="/tmp/atlas-070-hardware-ui.png")
            page.locator(".modal-heading button").click()
            if renderer == "webgl":
                page.evaluate(
                    "document.querySelector('.scene-gpu').getContext('webgl').getExtension('WEBGL_lose_context').loseContext()"
                )
                expect(page.locator(".scene-gpu")).to_have_attribute("data-backend", "fallback")
                expect(page.locator(".scene-layer")).not_to_have_class(re.compile("gpu-active"))
                expect(page.locator(".scene-tiles img").first).to_be_visible()
            page.screenshot(path="/tmp/atlas-070-precision-ui.png")
            assert not errors, errors
            assert not external, external
            print(
                f"Precision UI passed: mask review/apply/reset, negative example, adaptation, honest missing metrics, index/approximate search, feedback, detailed CSV, hardware controls, renderer={renderer}, context-loss fallback, zero JS errors/external requests. Test embeddings only."
            )
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
