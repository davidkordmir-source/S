from __future__ import annotations

import io
import re
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

from .config import WMS_URL
from .errors import UserError
from .imagery import PDOK


class MapTiles:
    """On-demand fixed-host tiles using the same verified proxy route as PDOK.

    No prefetching or bulk downloads. Cache freshness follows upstream headers;
    when no freshness header exists, use seven days per OSM tile policy.
    """

    def __init__(self, root: Path, provider: PDOK):
        self.root = root / "map-cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.layer_name: str | None = None
        self.lock = threading.Lock()

    def get(self, kind: str, z: int, x: int, y: int) -> tuple[bytes, str, int]:
        if kind not in ("street", "aerial") or not 0 <= z <= 20 or not 0 <= x < 2**z or not 0 <= y < 2**z:
            raise UserError("مختصات کاشی نقشه معتبر نیست.")
        if kind == "street" and z > 19:
            raise UserError("بزرگ‌نمایی نقشهٔ خیابانی بیش از حد مجاز است.")
        destination = self.root / f"{kind}-{z}-{x}-{y}.tile"
        expiry = destination.with_suffix(".expires")
        mime = "image/png" if kind == "street" else "image/jpeg"
        with self.lock:
            if destination.is_file() and expiry.is_file():
                try:
                    remaining = int(float(expiry.read_text()) - time.time())
                    if remaining > 0:
                        return destination.read_bytes(), mime, remaining
                except (ValueError, OSError):
                    pass
        if kind == "street":
            # Host and URL shape are constants, never user-controlled URLs.
            url, params = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", {}
        else:
            with self.lock:
                if self.layer_name is None:
                    self.layer_name = self.provider.layer()
            span = 40075016.68557849 / 2**z
            left, top = -20037508.342789244 + x * span, 20037508.342789244 - y * span
            url, params = (
                WMS_URL,
                {
                    "service": "WMS",
                    "version": "1.1.1",
                    "request": "GetMap",
                    "layers": self.layer_name,
                    "styles": "",
                    "srs": "EPSG:3857",
                    "bbox": f"{left},{top - span},{left + span},{top}",
                    "width": 256,
                    "height": 256,
                    "format": "image/jpeg",
                },
            )
        # Fetch through the provider's TLS-verifying, proxy-aware client. This
        # avoids teaching Qt/browser instances separate proxy credentials.
        try:
            with self.provider.client.stream("GET", url, params=params) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 2 * 1024 * 1024:
                        raise UserError("کاشی نقشه بیش از حد بزرگ است.")
                mime = response.headers.get("content-type", "").split(";", 1)[0]
                cache_control = response.headers.get("cache-control", "")
                maximum = re.search(r"(?:^|,)\s*max-age=(\d+)", cache_control)
                freshness = int(maximum.group(1)) if maximum else 7 * 24 * 3600
                if "no-store" in cache_control or "no-cache" in cache_control:
                    freshness = 0
        except UserError:
            raise
        except Exception as exc:
            raise UserError("دریافت نقشه ممکن نشد. اتصال شبکه را بررسی کنید.") from exc
        if mime not in ("image/png", "image/jpeg"):
            raise UserError("سرویس نقشه تصویر معتبری برنگرداند.")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.width > 1024 or image.height > 1024:
                    raise ValueError("oversized tile")
                image.verify()
        except Exception as exc:
            raise UserError("کاشی نقشه قابل‌خواندن نیست.") from exc
        if freshness > 0:
            with self.lock:
                with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as temporary:
                    temporary.write(content)
                    pending = Path(temporary.name)
                pending.replace(destination)
                expiry.write_text(str(time.time() + freshness))
                self._trim()
        return bytes(content), mime, freshness

    def _trim(self):
        files = list(self.root.glob("*.tile"))
        if len(files) > 512:
            for path in sorted(files, key=lambda file: file.stat().st_mtime)[: len(files) - 512]:
                path.unlink(missing_ok=True)
                path.with_suffix(".expires").unlink(missing_ok=True)
