from typing import List, Any

from py3r.point_tracking.core.model.pose_model import PoseModel
from py3r.point_tracking.core.types import PoseInstance


class MockPoseModel(PoseModel):
    def __init__(self, instances: List[PoseInstance]):
        self.instances = instances

    def _predict(self, _img: Any) -> List[PoseInstance]:
        return self.instances
