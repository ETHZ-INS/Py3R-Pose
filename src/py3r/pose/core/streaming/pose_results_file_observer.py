from pathlib import Path
from threading import Lock

from py3r.pose.core.serialization.dynamic_csv_writer import DynamicPoseCSVWriter

import reactivex as rx
from reactivex import Observer, Observable
from reactivex.abc import DisposableBase
from reactivex.disposable import Disposable

from py3r.pose.core.types import VideoFramePoses


class _PoseResultsWriterResource(Disposable):
    """Owns writer teardown. Idempotent & thread-safe."""
    def __init__(self, writer: DynamicPoseCSVWriter):
        super().__init__()
        self._writer = writer
        self._closed = False
        self._lock = Lock()

    def dispose(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._writer.close()
            except Exception:
                pass


class PoseResultsFileObserver(Observer):
    """
    RxPY Observer that writes pose results to a CSV file.

    - Exposes a single Future `result`:
        * set_result(None) on success (on_completed)
        * set_exception(exc) on error
        * canceled if disposed before completion
    - Thread-safe, idempotent cleanup, tolerant of late notifications.
    """

    def __init__(self, target_file: Path):
        super().__init__()
        self._target_file = Path(target_file)
        self._csv_writer = DynamicPoseCSVWriter(self._target_file)

    def using(self, upstream: rx.Observable[VideoFramePoses]):
        def resource_factory():
            # Create the teardown resource now; open in observable_factory so we can
            # translate open() failures into an observable error.
            return _PoseResultsWriterResource(self._csv_writer)

        def observable_factory(_res):
            return upstream

        return rx.using(resource_factory, observable_factory)

    def _on_next_core(self, pose_results):
        if self._csv_writer is None or self._csv_writer.closed:
            return

        try:
            self._csv_writer.write(pose_results)
        except Exception as e:
            self.on_error(e)
