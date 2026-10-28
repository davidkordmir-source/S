"""Browser wiring for optional one/many-view teaching; model quality is checked separately."""

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
from atlas.workshop import box_tuple, intersection_over_union  # noqa:E402


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
        scene, examples, truth, _ = make_teaching_scene()
        data = io.BytesIO()
        scene.save(data, "PNG")
        with sync_playwright() as p:
            chromium = os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium")
            browser = p.chromium.launch(headless=True, **({"executable_path": chromium} if chromium else {}))
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            external = []
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
            page.get_by_role("button", name="تطابق دقیق", exact=True).click()
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {"name": "scene.png", "mimeType": "image/png", "buffer": data.getvalue()}
            )
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=15000)
            page.get_by_test_id("reference-input").set_input_files(
                {"name": "view-a.png", "mimeType": "image/png", "buffer": examples[0]}
            )
            expect(page.locator(".teaching-gallery article")).to_have_count(1)
            group = page.get_by_role("combobox", name="هدف تصویری").input_value()
            slider = page.get_by_role("slider", name="حداقل شباهت", exact=True)
            slider.press("Home")
            for _ in range(30):
                slider.press("ArrowUp")
            expect(slider).to_have_value("0.7")

            def search():
                button = page.locator(".similar-footer .primary-button")
                expect(button).to_be_enabled(timeout=10000)
                button.click()
                expect(page.locator(".ranked-results .result-card").first).to_be_visible(timeout=60000)
                path = page.locator(".result-photo img").first.get_attribute("src")
                run = re.search(r"/runs/([^/]+)/", path).group(1)
                response = page.request.get("http://127.0.0.1:8766/api/workspace/runs/" + run)
                assert response.ok
                return response.json()

            def matches(run, target):
                return [
                    d
                    for d in run["detections"]
                    if intersection_over_union(box_tuple(d["bbox"], 768, 384), target) > 0.5
                ]

            first = search()
            assert matches(first, truth[0]) and not matches(first, truth[1])
            page.get_by_test_id("reference-input").set_input_files(
                [
                    {"name": "view-b.png", "mimeType": "image/png", "buffer": examples[1]},
                    {"name": "duplicate-b.png", "mimeType": "image/png", "buffer": examples[1]},
                ]
            )
            expect(page.locator(".teaching-gallery article")).to_have_count(2)
            expect(page.locator(".result-card")).to_have_count(0)
            expect(page.locator(".reference-preview b")).to_have_text("view-a.png")
            second = search()
            assert all(matches(second, target) for target in truth)
            b = matches(second, truth[1])[0]
            page.locator(".result-photo").nth(b["index"]).click()
            expect(page.locator(".reference-comparison")).to_be_visible()
            assert b["matched_reference"] in page.get_by_role(
                "img", name="نمونهٔ اصلی برای مقایسه"
            ).get_attribute("src")
            page.get_by_role("checkbox", name="استفاده از نمونه view-b.png", exact=True).uncheck()
            expect(page.locator(".result-card")).to_have_count(0)
            disabled = search()
            assert not matches(disabled, truth[1])
            page.get_by_role("checkbox", name="استفاده از نمونه view-b.png", exact=True).check()
            expect(page.locator(".result-card")).to_have_count(0)
            page.get_by_role("textbox", name="نام هدف تصویری جدید", exact=True).fill("هدف جداگانه")
            page.get_by_role("button", name="ساخت هدف تصویری", exact=True).click()
            expect(page.locator(".teaching-gallery article")).to_have_count(0)
            expect(page.locator(".similar-footer .primary-button")).to_be_disabled()
            page.get_by_role("combobox", name="هدف تصویری").select_option(group)
            expect(page.locator(".teaching-gallery article")).to_have_count(2)
            expect(
                page.get_by_role("checkbox", name="استفاده از نمونه view-b.png", exact=True)
            ).to_be_checked()
            page.reload()
            expect(page.get_by_role("combobox", name="هدف تصویری")).to_have_value(group)
            expect(page.locator(".teaching-gallery article")).to_have_count(2)
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(400)
            page.screenshot(path="/tmp/atlas-060-teaching-ui.png")
            assert not errors, errors
            assert not external, external
            browser.close()
        print(
            "Teaching UI passed: one view search, optional multi-file views, duplicate prevention, best-view comparison, disable/re-enable, isolated profiles, zero-view guard, active profile persistence, zero external requests/JS errors. Test embeddings only; actual DINO quality is a separate packaged check."
        )
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
