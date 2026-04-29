"""Observability helpers for background memory ops.

The audit found 23 "Memory generation triggered in background" entries with
ZERO matching completion logs. This module fixes that: every fire-and-forget
thread is wrapped to log start, completion, and latency.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@contextmanager
def timed(label: str, *, level: int = logging.INFO) -> Iterator[dict]:
    """Context manager that logs label start + completion (or failure) with elapsed ms.

    Usage:
        with timed("session-end generate") as ctx:
            ctx["events"] = len(events)
            do_work()
    """
    ctx: dict = {}
    start = time.monotonic()
    logger.log(level, "gmb-bg start: %s", label)
    try:
        yield ctx
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning("gmb-bg fail: %s after %.0fms (%s: %s) ctx=%s",
                       label, elapsed_ms, type(e).__name__, e, ctx)
        raise
    else:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.log(level, "gmb-bg done: %s in %.0fms ctx=%s", label, elapsed_ms, ctx)


def named_thread(target, *, name: str, args=(), kwargs=None) -> threading.Thread:
    """Spawn a daemon thread with a stable, greppable name."""
    t = threading.Thread(target=target, args=args, kwargs=kwargs or {}, daemon=True, name=name)
    t.start()
    return t


class ScopeDriftDetector:
    """Tracks the active scope across the process lifetime and warns on drift.

    The audit found 3 different user_ids and 3 different engines for the same
    human across 4 days. Drift creates orphaned memory clusters. This detector
    logs once per (user, app, engine) tuple and SCREAMS when it changes.
    """

    def __init__(self) -> None:
        self._observed: Optional[tuple] = None
        self._lock = threading.Lock()

    def record(self, *, user_id: str, app_name: str, engine_id: str) -> None:
        key = (user_id, app_name, engine_id)
        with self._lock:
            if self._observed is None:
                self._observed = key
                logger.info(
                    "gmb-scope: established scope user=%s app=%s engine=%s", *key
                )
                return
            if self._observed != key:
                logger.warning(
                    "gmb-scope DRIFT detected! was %s now %s — memories will split.",
                    self._observed, key,
                )
                self._observed = key
