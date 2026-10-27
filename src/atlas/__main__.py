from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from .config import data_directory
from .offline_server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Scout Windows desktop building search")
    parser.add_argument("--serve", action="store_true", help="Run only the loopback UI/API for development")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Check the desktop document and session, then exit"
    )
    parser.add_argument(
        "--ai-smoke-test", action="store_true", help="Validate real models and imagery, then exit"
    )
    parser.add_argument(
        "--formats-smoke-test",
        action="store_true",
        help="Validate bundled image decoders and actual ECW files, then exit",
    )
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: choose an unused port)")
    parser.add_argument(
        "--large-files-smoke-test", action="store_true", help="Check sparse BigTIFF addressing beyond 1 TiB"
    )
    parser.add_argument(
        "--ecw-index-smoke-test",
        action="store_true",
        help="Validate ECW directories exceeding the former index cap",
    )
    parser.add_argument(
        "--similarity-smoke-test",
        action="store_true",
        help="Validate screenshot search with real image embeddings",
    )
    args = parser.parse_args()
    root = data_directory()
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            RotatingFileHandler(root / "atlas.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        ],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.ecw_index_smoke_test:
        from .ecw_validation import validate_ecw_directories

        validate_ecw_directories(root)
        return
    if args.similarity_smoke_test:
        from .similarity_validation import validate_similarity

        validate_similarity(root)
        return
    if args.formats_smoke_test:
        from .validation import validate_packaged_formats

        validate_packaged_formats(root)
        return
    if args.large_files_smoke_test:
        from .large_validation import validate_large_files

        validate_large_files(root)
        return
    if args.ai_smoke_test:
        from .validation import validate_packaged_ai

        validate_packaged_ai(root)
        return
    app = create_app(root)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(128)
    port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/?token={app.state.token}"
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, access_log=False, log_config=None, timeout_graceful_shutdown=3
        )
    )
    if args.serve:
        # A per-launch local session URL, never a reusable credential/API key.
        print(f"Atlas development session: {url}", flush=True)
        server.run(sockets=[listener])
        return
    try:
        from PySide6.QtCore import QFile, QIODevice, QTimer, QUrl
        from PySide6.QtGui import QColor
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineCore import (
            QWebEnginePage,
            QWebEngineProfile,
            QWebEngineSettings,
            QWebEngineUrlRequestInterceptor,
        )
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow
    except ImportError as exc:
        raise SystemExit("Install the desktop dependencies first: pip install -e .") from exc

    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self, destination, navigation_type, is_main_frame):
            if destination.scheme() == "blob" and destination.toString().startswith(
                f"blob:http://127.0.0.1:{port}/"
            ):
                return True
            if (
                destination.scheme() == "http"
                and destination.host() == "127.0.0.1"
                and destination.port() == port
            ):
                return True
            return False

    class OfflineInterceptor(QWebEngineUrlRequestInterceptor):
        def interceptRequest(self, information):
            destination = information.requestUrl()
            local = (
                destination.scheme() == "http"
                and destination.host() == "127.0.0.1"
                and destination.port() == port
            )
            if not local and destination.scheme() not in ("data", "blob", "about"):
                information.block(True)

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Atlas Scout")
    application.setOrganizationName("AtlasScout")
    window = QMainWindow()
    window.setWindowTitle("اطلس آفلاین | آموزش و جست‌وجو در تصویر")
    window.resize(1440, 940)
    window.setMinimumSize(1080, 760)
    view = QWebEngineView(window)
    # Off-the-record profile: session cookies do not survive app restarts.
    profile = QWebEngineProfile(view)
    interceptor = OfflineInterceptor(profile)
    profile.setUrlRequestInterceptor(interceptor)

    def save_download(request):
        filename, _ = QFileDialog.getSaveFileName(
            window,
            "ذخیرهٔ خروجی",
            str(Path.home() / "Downloads" / Path(request.suggestedFileName()).name),
            "All files (*);;PNG images (*.png);;CSV (*.csv);;Training ZIP (*.zip)",
        )
        if filename:
            selected = Path(filename)
            request.setDownloadDirectory(str(selected.parent))
            request.setDownloadFileName(selected.name)
            request.accept()
        else:
            request.cancel()

    profile.downloadRequested.connect(save_download)
    page = LocalPage(profile, view)
    page.setBackgroundColor(QColor("#101814"))
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)
    view.setPage(page)
    from .desktop_bridge import FileBridge

    script = QFile(":/qtwebchannel/qwebchannel.js")
    if not script.open(QIODevice.OpenModeFlag.ReadOnly):
        raise RuntimeError("The native file selection channel is unavailable")
    app.state.desktop_channel_script = bytes(script.readAll())
    script.close()
    bridge = FileBridge(app.state.services, window)
    channel = QWebChannel(page)
    channel.registerObject("atlasFiles", bridge)
    page.setWebChannel(channel)
    window.setCentralWidget(view)
    smoke_passed = []

    if args.smoke_test:

        def inspect_document():
            def received(value):
                passed = value is True
                smoke_passed.append(passed)
                print("Desktop document smoke: " + ("PASS" if passed else "FAIL"), flush=True)
                application.quit()

            page.runJavaScript(
                "Boolean(window.__atlasNativeReady && document.querySelector('h1') && document.querySelector('.scene-viewport') && !document.querySelector('.notice.error'))",
                received,
            )

        def document_loaded(ok):
            if ok:
                QTimer.singleShot(2500, inspect_document)

        view.loadFinished.connect(document_loaded)
        QTimer.singleShot(20000, application.quit)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    attempts = 0

    def load_when_ready():
        nonlocal attempts
        attempts += 1
        if server.started:
            view.setUrl(QUrl(url))
        elif attempts < 100:
            QTimer.singleShot(100, load_when_ready)
        else:
            logging.error("Local server did not start")
            application.quit()

    QTimer.singleShot(0, load_when_ready)
    window.show()
    application.exec()
    server.should_exit = True
    thread.join(timeout=5)
    if args.smoke_test and smoke_passed != [True]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
