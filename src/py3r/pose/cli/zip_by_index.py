from collections import deque
from threading import RLock
from typing import Callable, TypeVar, Any

import reactivex as rx
from reactivex import Observable
from reactivex.disposable import CompositeDisposable

T = TypeVar("T")


def zip_by_index(
    *observables: Observable,
    fill_value=None,
    index: Callable[[Any], Any] = lambda x: x.index,
) -> Observable:
    """
    Zip multiple observables by monotonically increasing item index.

    Each input observable must emit items whose `index(item)` values are in
    strictly increasing order.

    For every index seen in any input, emit one tuple of length N:
      - the matching item from each observable if present at that index
      - otherwise `fill_value`

    Example:
        [0, 1, 3, 4, 5]
        [1, 2, 3, 4, 5]

        ->

        [(a0, None),
         (a1, b1),
         (None, b2),
         (a3, b3),
         (a4, b4),
         (a5, b5)]
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
                # Smallest currently visible index across all queues
                visible_indices = [
                    index(q[0]) for q in queues if q
                ]

                if not visible_indices:
                    if all(done):
                        observer.on_completed()
                    return

                current_index = min(visible_indices)

                # We can emit current_index only if every source is in one of:
                # - has head at current_index
                # - has head > current_index
                # - is completed
                # If a source is empty and not completed, we must wait because it
                # might still emit current_index later.
                row = []
                to_pop = []

                for i in range(n):
                    q = queues[i]

                    if q:
                        head = q[0]
                        head_index = index(head)

                        if head_index == current_index:
                            row.append(head)
                            to_pop.append(i)
                        elif head_index > current_index:
                            row.append(fill_value)
                        else:
                            observer.on_error(
                                ValueError(
                                    f"Observable {i} emitted out-of-order index "
                                    f"{head_index} after {current_index}"
                                )
                            )
                            return
                    else:
                        if done[i]:
                            row.append(fill_value)
                        else:
                            return  # Need more information from this source

                for i in to_pop:
                    queues[i].popleft()

                observer.on_next(tuple(row))

                if all(done) and all(not q for q in queues):
                    observer.on_completed()
                    return

        def make_on_next(i):
            def _on_next(value):
                with lock:
                    q = queues[i]
                    new_index = index(value)

                    if q and index(q[-1]) >= new_index:
                        observer.on_error(
                            ValueError(
                                f"Observable {i} emitted non-increasing index "
                                f"{new_index}"
                            )
                        )
                        return

                    q.append(value)
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
