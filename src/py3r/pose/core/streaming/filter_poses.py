from typing import List,  Callable

import reactivex as rx
import reactivex.operators as ops
from reactivex import Observable

from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.types import PoseInstance


def filter_poses(pose_filter: IPoseFilter, filter_all: bool = False) -> Callable[[Observable[List[PoseInstance]]], Observable[List[PoseInstance]]]:
    def _filter(poses: List[PoseInstance]) -> List[PoseInstance]:
        return pose_filter.filter(poses)

    def _filter_all(pose_results_list: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        return pose_filter.filter_all(pose_results_list)

    def _op(upstream):
        if filter_all:
            return upstream.pipe(
                ops.to_list(),
                ops.map(_filter_all),
                ops.flat_map(lambda x: rx.from_iterable(x))
            )

        return upstream.pipe(
            ops.map(_filter)
        )

    return _op
