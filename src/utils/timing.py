from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timed_step(timings: Dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        timings[key] = end - start
