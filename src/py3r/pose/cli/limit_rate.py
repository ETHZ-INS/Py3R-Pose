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
            next_due = time.monotonic()

            def dispose_once() -> None:
                nonlocal disposed
                with lock:
                    disposed = True

            def on_next(item: T) -> None:
                nonlocal next_due

                while True:
                    with lock:
                        if disposed:
                            return

                        now = time.monotonic()
                        wait = next_due - now
                        if wait <= 0:
                            # Reserve the next slot before calling downstream.
                            next_due = max(next_due + period, now + period)
                            break

                    # Sleep outside the lock so dispose() is not blocked.
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
