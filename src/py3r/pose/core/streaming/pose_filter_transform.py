from typing import List, Union

import reactivex as rx
import reactivex.operators as ops

from py3r.pose.core.filtering.pose_filter import PoseFilter
from py3r.pose.core.types import Poses, HasPoses, PoseInstance


class PoseFilterTransform:
    """Callable object: upstream Observable -> downstream Observable."""
    def __init__(self, pose_filter: PoseFilter, filter_all: bool = False):
        self.pose_filter = pose_filter
        self.filter_all = filter_all

    def _filter(self, pose_results: HasPoses) -> Poses:
        return self.pose_filter.filter(pose_results)

    def _filter_all(self, pose_results_list: List[HasPoses]) -> List[Poses]:
        return [self._filter(pr) for pr in pose_results_list]

    def __call__(self, poses: rx.Observable[Union[HasPoses, List[PoseInstance]]]) -> rx.Observable[Poses]:
        if self.filter_all:
            return poses.pipe(
                ops.map(lambda x: Poses(x) if isinstance(x, list) else x),
                ops.to_list(),
                ops.map(self._filter_all),
                ops.flat_map(lambda x: rx.from_iterable(x))
            )

        return poses.pipe(
            ops.map(lambda x: Poses(x) if isinstance(x, list) else x),
            ops.map(self._filter)
        )
