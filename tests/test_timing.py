from __future__ import annotations

import time

from src.utils.timing import timed_step


def test_timed_step_records_elapsed_time(monkeypatch):
    timings: dict[str, float] = {}

    def fake_perf_counter_factory():
        counter = 100.0

        def fake_perf_counter():
            nonlocal counter
            counter += 0.25
            return counter

        return fake_perf_counter

    fake_counter = fake_perf_counter_factory()
    monkeypatch.setattr(time, "perf_counter", fake_counter)

    with timed_step(timings, "sample"):
        pass

    assert timings["sample"] == 0.25
