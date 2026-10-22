from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .errors import Cancelled, UserError

logger = logging.getLogger(__name__)


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.status = "queued"
        self.message = "در صف پردازش"
        self.progress = 0
        self.result: Any = None
        self.error: str | None = None
        self.cancel_event = threading.Event()
        self.created = time.time()
        self.lock = threading.Lock()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def update(self, progress: float, message: str) -> None:
        self.check()
        with self.lock:
            self.progress = min(99, max(0, round(progress)))
            self.message = message

    def view(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "message": self.message,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
            }


class Jobs:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def submit(self, kind: str, task: Callable[[Job], Any]) -> dict:
        with self.lock:
            if any(j.status in ("queued", "running") for j in self.jobs.values()):
                raise UserError("یک عملیات در حال اجراست. تا پایان آن صبر کنید یا آن را لغو کنید.")
            # Retain only a bounded number of completed jobs.
            if len(self.jobs) >= 30:
                self.jobs.pop(next(iter(self.jobs)))
            job = Job(kind)
            self.jobs[job.id] = job
        self.pool.submit(self._run, job, task)
        return job.view()

    @staticmethod
    def _run(job: Job, task: Callable[[Job], Any]) -> None:
        try:
            job.check()
            with job.lock:
                job.status = "running"
            result = task(job)
            job.check()
            with job.lock:
                job.result = result
                job.status, job.progress, job.message = "done", 100, "عملیات تکمیل شد"
        except Cancelled:
            with job.lock:
                job.status, job.message = "cancelled", "عملیات لغو شد"
        except UserError as exc:
            with job.lock:
                job.status, job.error = "failed", str(exc)
        except Exception:
            logger.exception("Job %s failed", job.id)
            with job.lock:
                job.status = "failed"
                job.error = "پردازش کامل نشد. گزارش خطای برنامه را بررسی کنید و دوباره تلاش کنید."

    def get(self, id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(id)

    def close(self) -> None:
        with self.lock:
            for job in self.jobs.values():
                job.cancel_event.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
