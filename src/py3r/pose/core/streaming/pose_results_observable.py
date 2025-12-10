import time

import reactivex as rx
from reactivex.disposable import Disposable
from reactivex.scheduler import CurrentThreadScheduler

from py3r.pose.core.serialization.csv_reader import PoseCSVReader
from py3r.pose.core.types import VideoFramePoses


def pose_results_observable(src: PoseCSVReader, scheduler: rx.abc.SchedulerBase = None) -> rx.Observable[VideoFramePoses]:
    """
    Create an Observable that:
      - opens the VideoSource on subscribe (on `scheduler`)
      - reads frames on `scheduler` via a recursive scheduled action
      - closes the VideoSource on dispose/completion (on `scheduler`)
    """

    def resource_factory():
        return Disposable(lambda: src.close())

    def observable_factory(_res):
        def _subscribe(observer, scheduler_=None):
            _scheduler = scheduler or scheduler_ or CurrentThreadScheduler.singleton()
            cancelled = [False]

            time.sleep(1)

            def tick(_, __=None):
                if cancelled[0]:
                    return

                try:
                    frame_poses = src.read()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    observer.on_error(e)
                    return

                if frame_poses is None:
                    # EOF/no frame
                    observer.on_completed()
                    return

                observer.on_next(frame_poses)
                _scheduler.schedule(tick)

            # start the loop on the scheduler
            _scheduler.schedule(tick)

            # cooperative cancellation; close happens via the resource's dispose()
            def _cancel():
                cancelled[0] = True

            return Disposable(_cancel)

        return rx.create(_subscribe)
    return rx.using(resource_factory, observable_factory)
