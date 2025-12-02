from typing import Tuple, Callable, Union

import numpy as np
import reactivex as rx
import reactivex.operators as ops

from py3r.pose.core.types import HasPoses
from py3r.media.types import HasImage, Image
from py3r.pose.core.visualization.pose_renderer import PoseRenderer


class PoseRenderTransform:
    def __init__(self, pose_renderer: PoseRenderer):
        self._pose_renderer = pose_renderer

    def _visualize(self, pair: Tuple[HasPoses, np.ndarray]) -> Image:
        try:
            vis_img = self._pose_renderer.render(pair[1], pair[0].instances)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        return Image(vis_img)

    def __call__(self, images: rx.Observable[Union[HasImage, np.ndarray]]) -> Callable[[rx.Observable[HasPoses]], rx.Observable[Image]]:
        def inner(poses: rx.Observable) -> rx.Observable:
            imgs = images.pipe(ops.map(lambda x: x if isinstance(x, np.ndarray) else x.img))
            return poses.pipe(
                ops.zip(imgs),
                ops.map(self._visualize)
            )
        return inner
