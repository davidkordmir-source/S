"""Optional CUDA/DirectML acceleration with explicit, observable CPU fallback."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .errors import UserError

logger = logging.getLogger(__name__)


def windows_adapters():
    if os.name != "nt":
        return []
    import ctypes
    import uuid
    from ctypes import wintypes

    class Luid(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class Description(ctypes.Structure):
        _fields_ = [
            ("name", wintypes.WCHAR * 128),
            ("vendor", wintypes.UINT),
            ("device", wintypes.UINT),
            ("subsystem", wintypes.UINT),
            ("revision", wintypes.UINT),
            ("video", ctypes.c_size_t),
            ("system", ctypes.c_size_t),
            ("shared", ctypes.c_size_t),
            ("luid", Luid),
            ("flags", wintypes.UINT),
        ]

    factory = ctypes.c_void_p()
    dxgi = ctypes.WinDLL("dxgi.dll")
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    iid = ctypes.create_string_buffer(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)

    def method(pointer, index, result, *args):
        vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(vtable[index])

    result = []
    if create(iid, ctypes.byref(factory)) != 0:
        return result
    try:
        for index in range(16):
            adapter = ctypes.c_void_p()
            if (
                method(factory, 12, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(
                    factory, index, ctypes.byref(adapter)
                )
                != 0
            ):
                break
            try:
                desc = Description()
                if (
                    method(adapter, 10, ctypes.c_long, ctypes.POINTER(Description))(
                        adapter, ctypes.byref(desc)
                    )
                    == 0
                    and not desc.flags & 2
                ):
                    result.append(
                        dict(
                            id=index,
                            name=desc.name,
                            memory=int(desc.video + desc.shared),
                            dedicated=int(desc.video),
                        )
                    )
            finally:
                method(adapter, 2, wintypes.ULONG)(adapter)
    finally:
        method(factory, 2, wintypes.ULONG)(factory)
    return sorted(result, key=lambda adapter: adapter["dedicated"], reverse=True)


class Compute:
    def __init__(self, root: Path, graph: Path):
        self.path = root / "compute.json"
        self.graph = graph
        self.preference = "auto"
        self.effective = "CPU"
        self.reason = ""
        self.session = None
        self._gpu_verified = False
        self._dml_failed = False
        self.performance = "maximum"
        try:
            saved = json.loads(self.path.read_text())
            self.preference = saved["device"] if saved.get("device") in ("auto", "cpu", "gpu") else "auto"
            self.performance = (
                saved.get("performance", "maximum")
                if saved.get("performance", "maximum") in ("balanced", "maximum")
                else "maximum"
            )
        except (OSError, ValueError, KeyError):
            pass
        try:
            self.adapters = windows_adapters()
        except Exception:
            logger.exception("GPU adapter enumeration failed")
            self.adapters = []

    def capabilities(self):
        import torch

        cuda = bool(torch.cuda.is_available())
        dml = False
        if self.adapters and self.graph.is_file():
            try:
                import onnxruntime as ort

                ort.disable_telemetry_events()
                dml = "DmlExecutionProvider" in ort.get_available_providers()
            except ImportError:
                pass
        return cuda, dml

    def status(self):
        cuda, dml = self.capabilities()
        return dict(
            preference=self.preference,
            performance=self.performance,
            cpu_threads=self.cpu_threads(),
            device=self.effective,
            gpu_available=cuda or dml,
            adapters=self.adapters,
            cuda=cuda,
            directml=dml,
            fallback_reason=self.reason,
        )

    def cpu_threads(self):
        count = os.cpu_count() or 1
        try:
            quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
            if quota != "max":
                count = min(count, max(1, (int(quota) + int(period) - 1) // int(period)))
        except (OSError, ValueError):
            pass
        usable = max(1, count - 1)
        return usable if self.performance == "maximum" else min(4, usable)

    def configure(self, device, performance=None):
        if device not in ("auto", "cpu", "gpu"):
            raise UserError("حالت پردازش معتبر نیست.")
        if device == "gpu" and not any(self.capabilities()):
            raise UserError(
                "کارت گرافیک سازگار در این دستگاه پیدا نشد؛ حالت پردازنده یا خودکار را انتخاب کن."
            )
        if performance not in (None, "balanced", "maximum"):
            raise UserError("حالت کارایی معتبر نیست.")
        self.preference = device
        self.performance = performance or self.performance
        self.effective = "CPU"
        self.reason = ""
        self.session = None
        self._gpu_verified = False
        self._dml_failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(device=device, performance=self.performance)))
        temporary.replace(self.path)
        return self.status()

    def torch_device(self):
        if self.preference != "cpu" and self.capabilities()[0]:
            self.effective = "CUDA"
            return "cuda"
        return "cpu"

    def details(self, pixels):
        if self.preference == "auto" and self._dml_failed:
            return None
        if self.preference == "cpu" or self.capabilities()[0] or not self.capabilities()[1]:
            return None
        try:
            import onnxruntime as ort

            ort.disable_telemetry_events()
            if self.session is None:
                options = ort.SessionOptions()
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.intra_op_num_threads = self.cpu_threads()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                options.enable_profiling = True
                options.profile_file_prefix = str(self.path.parent / "gpu-probe")
                self.session = ort.InferenceSession(
                    str(self.graph),
                    sess_options=options,
                    providers=[
                        ("DmlExecutionProvider", {"device_id": self.adapters[0]["id"]}),
                        "CPUExecutionProvider",
                    ],
                )
                self.session.disable_fallback()
                if "DmlExecutionProvider" not in self.session.get_providers():
                    raise RuntimeError("DirectML provider did not activate")
            values = self.session.run(["tokens"], {"pixels": pixels})[0]
            if not getattr(self, "_gpu_verified", False):
                profile = Path(self.session.end_profiling())
                try:
                    events = json.loads(profile.read_text(encoding="utf-8"))
                    if not any(e.get("args", {}).get("provider") == "DmlExecutionProvider" for e in events):
                        raise RuntimeError("The DINO graph did not execute any DirectML nodes")
                finally:
                    profile.unlink(missing_ok=True)
                self._gpu_verified = True
            self.effective = "DirectML + CPU (DINOv2)"
            self.reason = ""
            return values
        except Exception as exc:
            logger.exception("GPU feature execution failed")
            self.session = None
            if self.preference == "gpu":
                raise UserError(
                    "پردازش کارت گرافیک فعال نشد؛ در تنظیمات، حالت خودکار یا پردازنده را انتخاب کن."
                ) from exc
            self.reason = "کارت گرافیک فعال نشد؛ پردازش با CPU ادامه پیدا کرد."
            self.effective = "CPU"
            self._dml_failed = True  # Preserve adapters and allow an explicit settings retry.
            return None
