from typing import Tuple, Callable, List

import numpy as np
import reactivex.operators as ops
from reactivex import Observable

from py3r.pose.core.types import PoseInstance

from py3r.pose.core.visualization.abc.pose_renderer import IPoseRenderer


def render_poses(pose_renderer: IPoseRenderer) -> Callable[[Observable[Tuple[np.ndarray, List[PoseInstance]]]], Observable[np.ndarray]]:
    def _render(pair: Tuple[np.ndarray, List[PoseInstance]]) -> np.ndarray:
        return pose_renderer.render(pair[0], pair[1])

    def _op(upstream):
        return upstream.pipe(
            ops.map(_render)
        )

    return _op
