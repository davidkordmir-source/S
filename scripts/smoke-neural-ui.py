"""Check neural search and editable/deletable targets/chat with explicit doubles."""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "tests"))
from offline_support import scene_bytes  # noqa: E402


def main():
    executable = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    server = subprocess.Popen([str(executable), "tests/ui_server.py"], cwd=ROOT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8766/", timeout=1).close()
                break
            except urllib.error.URLError:
                if server.poll() is not None:
                    raise RuntimeError("UI server failed") from None
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium"),
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://127.0.0.1:8766/?token=ui-test-only-session")
            page.locator('input[type=file]:not([multiple])[accept^=".png"]').set_input_files(
                {"name": "scene.png", "mimeType": "image/png", "buffer": scene_bytes()}
            )
            expect(page.locator(".scene-full.loaded")).to_be_visible(timeout=15000)
            expect(page.get_by_role("button", name="تشخیص هم‌نوع‌ها", exact=True)).to_have_class("active")
            expect(page.get_by_role("spinbutton", name="عرض نمونه روی تصویر")).to_have_count(0)
            page.get_by_role("textbox", name="خواستهٔ جست‌وجو", exact=True).fill("درخواست آزاد بدون عکس نمونه")
            search = page.locator(".similar-footer .primary-button")
            expect(search).to_be_enabled(timeout=10000)
            search.click()
            expect(page.locator(".result-card").first).to_be_visible(timeout=20000)
            page.get_by_role("textbox", name="نام هدف تصویری جدید").fill("متن اشتباه")
            page.get_by_role("button", name="ساخت هدف تصویری", exact=True).click()
            expect(page.get_by_role("textbox", name="نام هدف تصویری", exact=True)).to_have_value("متن اشتباه")
            page.get_by_role("textbox", name="خواستهٔ جست‌وجو", exact=True).fill("توضیح ذخیره‌شده")
            page.get_by_role("button", name="ذخیرهٔ توضیحات هدف", exact=True).click()
            expect(page.get_by_role("textbox", name="خواستهٔ جست‌وجو", exact=True)).to_have_value(
                "توضیح ذخیره‌شده"
            )
            page.get_by_role("button", name="پاک‌کردن توضیحات", exact=True).click()
            expect(page.get_by_role("textbox", name="خواستهٔ جست‌وجو", exact=True)).to_have_value("")
            page.get_by_text("گفت‌وگو با هوش مصنوعی آفلاین", exact=True).click()
            chat = page.get_by_role("textbox", name="پیام به هوش مصنوعی آفلاین", exact=True)
            chat.fill("خواستهٔ نامحدود آزمایشی")
            page.get_by_role("button", name="بررسی خواسته و نمونه‌ها", exact=True).click()
            expect(page.locator(".assistant-message")).to_have_count(2, timeout=15000)
            page.get_by_role("button", name="استفاده از این درخواست برای جست‌وجو", exact=True).click()
            expect(page.get_by_role("textbox", name="خواستهٔ جست‌وجو", exact=True)).to_have_value(
                "خواستهٔ نامحدود آزمایشی"
            )
            page.get_by_role("button", name="ویرایش پیام", exact=True).click()
            page.get_by_role("textbox", name="ویرایش پیام", exact=True).fill("اصلاح پیام")
            page.locator(".assistant-message.user").get_by_role("button", name="ذخیره", exact=True).click()
            expect(page.locator(".assistant-message.user p")).to_have_text("اصلاح پیام")
            page.get_by_role("button", name="پاک‌کردن گفتگو", exact=True).click()
            expect(page.locator(".assistant-message")).to_have_count(0)
            page.get_by_role("button", name="حذف هدف تصویری", exact=True).click()
            expect(page.get_by_role("combobox", name="هدف تصویری", exact=True)).to_have_count(0)
            page.get_by_role("button", name="بازگرداندن هدف", exact=True).click()
            expect(page.get_by_role("textbox", name="نام هدف تصویری", exact=True)).to_have_value("متن اشتباه")
            assert not errors, errors
            page.screenshot(path="/workspace/scratch/atlas080-neural-ui.png", full_page=True)
            browser.close()
            print(
                "Neural UI wiring passed: default instance mode, text-only search, editable/clearable descriptions, chat edit/delete, target delete/restore. Explicit doubles, not an accuracy test."
            )
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
