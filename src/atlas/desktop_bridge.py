"""Native file selection: the web document cannot supply arbitrary disk paths."""

from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QFileDialog

from .errors import UserError


class FileBridge(QObject):
    def __init__(self, services, window):
        super().__init__(window)
        self.services, self.window = services, window

    @Slot(bool, result="QVariantMap")
    def chooseScene(self, copy=False):
        try:
            self.services.idle()
            selected, _ = QFileDialog.getOpenFileName(
                self.window,
                "انتخاب تصویر ماهواره‌ای از دیسک",
                "",
                "Satellite images (*.ecw *.tif *.tiff *.jp2 *.j2k *.jpg *.jpeg *.png *.img *.ntf *.nitf *.pix *.heic *.heif *.avif *.webp *.bmp);;All files (*)",
            )
            if not selected:
                return {"cancelled": True}
            path = Path(selected)
            return {
                "job": self.services.jobs.submit(
                    "import-scene",
                    lambda job: self.services.workshop.import_scene_file(path, path.name, job, copy=copy),
                )
            }
        except UserError as exc:
            return {"error": str(exc)}
