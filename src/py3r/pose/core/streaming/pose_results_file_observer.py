from pathlib import Path
from concurrent.futures import Future
from threading import Lock

from reactivex.abc import DisposableBase

from py3r.pose.core.serialization.dynamic_csv_writer import DynamicCSVWriter

from reactivex import Observer, Observable
from reactivex.scheduler import ThreadPoolScheduler

from py3r.pose.core.types import VideoFramePoses


class PoseResultsFileObserver(Observer):
    """
    RxPY Observer that writes pose results to a CSV file.

    - Exposes a single Future `result`:
        * set_result(None) on success (on_completed)
        * set_exception(exc) on error
        * canceled if disposed before completion
    - Thread-safe, idempotent cleanup, tolerant of late notifications.
    """

    def __init__(self, target_file: Path, scheduler=None):
        super().__init__()
        self.target_file = Path(target_file)
        self._worker = scheduler or ThreadPoolScheduler(1)

        self.csv_writer = DynamicCSVWriter(self.target_file)

        self.result: Future = Future()

        self._sub = None
        self._stopped = False
        self._lock = Lock()

    # ---------- Observer interface ----------
    def on_next(self, pose_results):
        if self._stopped or self.csv_writer is None:
            return
        try:
            self.csv_writer.write(pose_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._fail(e)

    def on_error(self, err):
        if self._stopped:
            return
        self._fail(err)

    def on_completed(self):
        if self._stopped:
            return
        self._succeed()

    # ---------- Wiring ----------
    def attach(self, upstream: Observable[VideoFramePoses]) -> DisposableBase:
        """
        Subscribe to an Observable and remember the disposable.
        All callbacks are delivered on `self._worker` (serialized).
        """
        self._sub = upstream.subscribe(self)
        return self._sub

    def dispose(self):
        """Early stop: unsubscribe, cleanup, and mark as canceled."""
        self._unsubscribe()
        self._cleanup_once()
        with self._lock:
            if not self.result.done():
                self.result.cancel()

    # ---------- Internals ----------
    def _unsubscribe(self):
        sub = self._sub
        self._sub = None
        if sub is not None:
            try:
                sub.dispose()
            except Exception:
                pass  # best effort

    def _cleanup_once(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self.csv_writer is not None:
                try:
                    self.csv_writer.close()
                finally:
                    self.csv_writer = None

    def _fail(self, exc: Exception):
        self._unsubscribe()
        self._cleanup_once()
        with self._lock:
            if not self.result.done():
                self.result.set_exception(exc)

    def _succeed(self):
        self._unsubscribe()
        self._cleanup_once()
        with self._lock:
            if not self.result.done():
                self.result.set_result(self.target_file)
