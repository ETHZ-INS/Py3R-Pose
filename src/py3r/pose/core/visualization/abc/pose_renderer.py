from typing import List, Protocol

import numpy as np

from py3r.pose.core.types.instance import PoseInstance


class IPoseRenderer(Protocol):
    def render(self, img: np.ndarray, poses: List[PoseInstance]) -> np.ndarray: ...
