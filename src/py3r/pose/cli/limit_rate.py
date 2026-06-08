from __future__ import annotations

from typing import Callable, TypeVar
import threading
import time

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import Disposable
from reactivex.abc import ObserverBase, DisposableBase

T = TypeVar("T")


def limit_rate(
    items_per_second: float,
) -> Callable[[Observable[T]], Observable[T]]:
    """
    Emit items at no more than `rate` items per second.`.

    This operator does not drop items. If upstream is faster, it blocks the
    calling thread in `on_next`, which creates backpressure only for
    synchronous/cooperative sources.
    """
    if items_per_second <= 0:
        raise ValueError("rate_hz must be > 0")

    period = 1.0 / items_per_second

    def _operator(source: Observable[T]) -> Observable[T]:
        def _subscribe(
            observer: ObserverBase[T],
            _scheduler=None,
        ) -> DisposableBase:
            lock = threading.RLock()
            disposed = False
            # Initialised lazily on the first item so that startup delays
            # (scheduler overhead, skipped frames, etc.) don't build up debt.
            next_due = None

            def dispose_once() -> None:
                nonlocal disposed
                with lock:
                    disposed = True

            def on_next(item: T) -> None:
                nonlocal next_due

                with lock:
                    if disposed:
                        return
                    # Lazy init: anchor the clock to the first item's arrival
                    # so any pre-pipeline delay doesn't build up catch-up debt.
                    if next_due is None:
                        next_due = time.monotonic() - period
                    next_due += period
                    due = next_due  # capture for use outside the lock

                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

                with lock:
                    if disposed:
                        return
                    observer.on_next(item)

            def on_error(err: Exception) -> None:
                dispose_once()
                observer.on_error(err)

            def on_completed() -> None:
                dispose_once()
                observer.on_completed()

            upstream = source.subscribe(
                on_next,
                on_error,
                on_completed,
            )

            def dispose() -> None:
                dispose_once()
                upstream.dispose()

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _operator
