import collections
import threading
from typing import TypeVar, Optional, Callable, Deque

import reactivex as rx
from reactivex.disposable import Disposable, SerialDisposable
from reactivex.scheduler import TimeoutScheduler

_T = TypeVar("_T")


def preview_pace(
    period: float = 1/30.0,
    max_buffer: int = 5,
    scheduler: Optional[rx.abc.SchedulerBase] = None,
) -> Callable[[rx.abc.ObservableBase[_T]], rx.Observable[_T]]:

    if period <= 0:
        raise ValueError("period must be > 0")
    if max_buffer <= 0:
        raise ValueError("max_buffer must be > 0")

    def _op(source: rx.abc.ObservableBase[_T]) -> rx.Observable[_T]:
        def _subscribe(observer, scheduler_=None) -> Disposable:
            sch = scheduler or scheduler_ or TimeoutScheduler.singleton()

            buf: Deque[_T] = collections.deque()
            lock = threading.Lock()

            done = False
            disposed = False
            err: Optional[Exception] = None

            def on_next(x: _T):
                nonlocal disposed
                if disposed:
                    return
                with lock:
                    if len(buf) >= max_buffer:
                        buf.popleft()
                    buf.append(x)

            def on_error(e: Exception):
                nonlocal done, err
                err = e
                done = True

            def on_completed():
                nonlocal done
                done = True

            src_disp = source.subscribe(
                on_next,
                on_error,
                on_completed,
                scheduler=sch,
            )

            tick_disp = SerialDisposable()

            def tick(_sch, _state=None):
                nonlocal disposed

                if disposed:
                    return Disposable()

                if err is not None:
                    observer.on_error(err)
                    return Disposable()

                to_emit: Optional[_T] = None
                with lock:
                    if buf:
                        to_emit = buf.popleft()

                if to_emit is not None:
                    observer.on_next(to_emit)

                with lock:
                    empty = not buf
                if done and empty:
                    observer.on_completed()
                    return Disposable()

                tick_disp.disposable = sch.schedule_relative(period, tick)
                return Disposable()

            # start ticking
            tick_disp.disposable = sch.schedule_relative(period, tick)

            def dispose():
                nonlocal disposed
                disposed = True
                src_disp.dispose()
                tick_disp.dispose()
                with lock:
                    buf.clear()

            return Disposable(dispose)

        return rx.create(_subscribe)

    return _op
