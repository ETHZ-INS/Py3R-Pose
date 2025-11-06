from typing import List

import numpy as np
import reactivex as rx
import reactivex.operators as ops

from py3r.point_tracking.core.types import Poses, HasImage
from py3r.point_tracking.core.model.pose_model import PoseModel


class PoseEstimationTransform:
    """Callable object: upstream Observable -> downstream Observable."""
    def __init__(self, model: PoseModel, batch_size: int = 1):
        self.model = model
        self.batch_size = batch_size

    # --------------------------------------------------------------
    def _predict_batch(self, batch: List[np.ndarray]):
        try:
            poses = self.model.predict_batch(batch)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        results = poses
        return results

    def __call__(self, upstream: rx.Observable[HasImage | np.ndarray]) -> rx.Observable[Poses]:
        return upstream.pipe(
            ops.map(lambda x: x if isinstance(x, np.ndarray) else x.img),
            ops.buffer_with_count(self.batch_size),
            ops.map(self._predict_batch),
            ops.flat_map(lambda x: rx.from_iterable(x))
        )
