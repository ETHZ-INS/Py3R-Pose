from typing import List

import numpy as np

from py3r.point_tracking.core.model.pose_model import PoseModel
from py3r.point_tracking.core.types import PoseInstance


class MockPoseModel(PoseModel):
    def __init__(self, instances: List[PoseInstance]):
        self.instances = instances

    def predict(self, _img: np.ndarray) -> List[PoseInstance]:
        return self.instances
