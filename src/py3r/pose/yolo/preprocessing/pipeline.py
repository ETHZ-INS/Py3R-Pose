from typing import Tuple, List, Protocol, Any, Sequence, Union

import numpy as np
from py3r.pose.core.types import PoseInstance
from py3r.pose.core.types import PosePoint


# ---------- Base API ----------

class Transform(Protocol):
    def forward(self, img: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Return (img2, meta) where meta holds what invert_points needs."""
        raise NotImplementedError

    def forward_batch(self, imgs: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[Any]]:
        """Return (imgs2, metas) where metas holds what invert_points needs."""
        out_imgs = []
        out_metas = []
        for img in imgs:
            img2, meta = self.forward(img)
            out_imgs.append(img2)
            out_metas.append(meta)
        return out_imgs, out_metas

    def invert_point(self, xy: Tuple[float, float], meta: Any) -> Tuple[float, float]:
        """Map points from transformed image back to the previous image coords."""
        raise NotImplementedError

    def invert_poses(self, poses: Sequence[PoseInstance], meta: Any) -> List[PoseInstance]:
        inverted_instances = []
        for instance in poses:
            box = instance.box
            box_p1, box_p2 = (box[0], box[1]), (box[2], box[3])
            box_p1 = self.invert_point(box_p1, meta)
            box_p2 = self.invert_point(box_p2, meta)
            box = (box_p1[0], box_p1[1], box_p2[0], box_p2[1])

            points = []
            for point in instance.points:
                point_xy = (point.x, point.y)
                point_xy = self.invert_point(point_xy, meta)
                points.append(PosePoint(point_xy[0], point_xy[1], point.conf))

            inverted_instances.append(PoseInstance(
                instance.id,
                instance.type,
                box,
                points,
                instance.conf
            ))

        return inverted_instances


class Pipeline(Transform):
    """Chain of transforms; stores metas and can invert points/boxes."""
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def forward(self, img: np.ndarray):
        metas = []
        x = img
        for t in self.transforms:
            x, m = t.forward(x)
            metas.append(m)
        return x, metas

    def invert_point(self, xy: Tuple[float, float], metas: List[Any]) -> Tuple[float, float]:
        """xy: (N,2) in final image coords -> original image coords"""
        out = xy
        for t, m in zip(reversed(self.transforms), reversed(metas)):
            out = t.invert_points(out, m)
        return out
