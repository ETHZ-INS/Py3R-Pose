from __future__ import annotations

from typing import Callable, Protocol

import cv2
import numpy as np
import reactivex as rx
from reactivex import Observable
from reactivex.abc import SchedulerBase, ObserverBase
from reactivex.disposable import Disposable, SerialDisposable


class IImageDisplay(Protocol):
    """Protocol for image display backends used by the display_image operator."""

    def setup(self) -> None:
        """Called once when the display is first set up (e.g. create a window)."""
        ...

    def display(self, img: np.ndarray) -> None:
        """Called for each frame to display it."""
        ...

    def is_open(self) -> bool:
        """Return True while the display is still active (e.g. window not closed)."""
        ...

    def teardown(self) -> None:
        """Called once when the display is torn down (e.g. destroy the window)."""
        ...


class OpenCVImageDisplay:
    """cv2-backed image display for production use."""

    def __init__(self, window_name: str, flags: int = cv2.WINDOW_AUTOSIZE, wait_ms: int = 1):
        self.window_name = window_name
        self.flags = flags
        self.wait_ms = wait_ms

    def setup(self) -> None:
        cv2.namedWindow(self.window_name, self.flags)

    def display(self, img: np.ndarray) -> None:
        cv2.imshow(self.window_name, img)
        cv2.waitKey(self.wait_ms)

    def is_open(self) -> bool:
        # treat <= 0 (and -1 on some platforms) as closed
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) > 0.5

    def teardown(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def display_image(
    display: IImageDisplay,
    *,
    scheduler: SchedulerBase,
) -> Callable[[Observable[np.ndarray]], Observable[np.ndarray]]:
    """
    Rx operator that passes frames through while displaying each one as a side effect.

    setup/display/teardown are all dispatched onto *scheduler*, so the display object
    is only ever touched from that one thread — no locking required.

    For tests, pass CurrentThreadScheduler() so that display calls execute
    synchronously on the calling thread without spawning a background thread.
    """

    def _operator(source: Observable[np.ndarray]) -> Observable[np.ndarray]:
        def _subscribe(
            observer: ObserverBase[np.ndarray],
            _subscribe_scheduler: SchedulerBase | None = None,
        ):
            disposed = False
            upstream = SerialDisposable()

            def do_setup(*_):
                if not disposed:
                    display.setup()

            scheduler.schedule(do_setup)

            def on_next(img: np.ndarray) -> None:
                def do_display(*_):
                    if not disposed:
                        display.display(img)

                scheduler.schedule(do_display)
                observer.on_next(img)

            def on_error(err: Exception) -> None:
                observer.on_error(err)

            def on_completed() -> None:
                observer.on_completed()

            upstream.disposable = source.subscribe(on_next, on_error, on_completed)

            def dispose() -> None:
                nonlocal disposed
                if disposed:
                    return
                disposed = True
                upstream.dispose()

                def do_teardown(*_):
                    try:
                        display.teardown()
                    except Exception:
                        pass

                scheduler.schedule(do_teardown)

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _operator
