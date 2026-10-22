import threading

import pytest

from atlas.errors import UserError
from atlas.jobs import Jobs


def test_only_one_active_job_and_cancellation():
    jobs = Jobs()
    started, proceed, finished = threading.Event(), threading.Event(), threading.Event()

    def task(job):
        started.set()
        proceed.wait(timeout=2)
        try:
            job.check()
        finally:
            finished.set()

    result = jobs.submit("test", task)
    assert started.wait(timeout=2)
    with pytest.raises(UserError):
        jobs.submit("second", task)
    job = jobs.get(result["id"])
    job.cancel_event.set()
    proceed.set()
    assert finished.wait(timeout=2)
    jobs.pool.shutdown(wait=True)
    assert job.view()["status"] == "cancelled"
