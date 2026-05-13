from collections import deque
from threading import RLock

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import CompositeDisposable


def zip_longest(*observables: Observable, fill_value=None):
    """
    Zip any number of observables, padding completed/empty positions with
    `fill_value` until all observables are exhausted.

    Similar to itertools.zip_longest.

    Example:
        zip_longest(obs1, obs2, obs3, fill_value=None)

    Emits tuples of length len(observables).
    """

    if not observables:
        return rx.empty()

    def _subscribe(observer, scheduler=None):
        n = len(observables)
        queues = [deque() for _ in range(n)]
        done = [False] * n
        lock = RLock()

        def try_emit():
            while True:
                row = []
                can_emit = True

                for i in range(n):
                    if queues[i]:
                        row.append(queues[i].popleft())
                    elif done[i]:
                        row.append(fill_value)
                    else:
                        can_emit = False
                        break

                if can_emit:
                    observer.on_next(tuple(row))

                    # If everybody is done and all queues are empty, we're finished.
                    if all(done) and all(not q for q in queues):
                        observer.on_completed()
                        return
                else:
                    # No full row available yet.
                    if all(done) and all(not q for q in queues):
                        observer.on_completed()
                        return
                    return

        def make_on_next(i):
            def _on_next(value):
                with lock:
                    queues[i].append(value)
                    try_emit()
            return _on_next

        def make_on_completed(i):
            def _on_completed():
                with lock:
                    done[i] = True
                    try_emit()
            return _on_completed

        subscriptions = [
            obs.subscribe(
                on_next=make_on_next(i),
                on_error=observer.on_error,
                on_completed=make_on_completed(i),
                scheduler=scheduler,
            )
            for i, obs in enumerate(observables)
        ]

        return CompositeDisposable(*subscriptions)

    return rx.create(_subscribe)
