from typing import List, Union, Callable, Any

import numpy as np
import reactivex as rx
import reactivex.operators as ops

from reactivex import Observable

from py3r.pose.core.model.abc.pose_model import IPoseModel
from py3r.pose.core.types import PoseInstance


def predict_poses(pose_model: IPoseModel, batch_size: int = 1) -> Callable[[Observable[np.ndarray]], Observable[List[PoseInstance]]]:
    def _predict_batch(batch: List[Any]) -> List[List[PoseInstance]]:
        return pose_model.predict_batch(batch)

    def _op(upstream):
        return upstream.pipe(
            ops.buffer_with_count(batch_size),
            ops.map(_predict_batch),
            ops.flat_map(lambda x: rx.from_iterable(x))
        )

    return _op
