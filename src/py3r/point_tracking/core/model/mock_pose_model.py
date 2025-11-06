from typing import List, Any

from py3r.point_tracking.core.model.pose_model import PoseModel
from py3r.point_tracking.core.types import PoseInstance, PoseInstanceType


class MockPoseModel(PoseModel):
    def __init__(self, instances: List[PoseInstance]):
        self.instances = instances

    def get_instance_types(self) -> List[PoseInstanceType]:
        unique_instance_types = {}
        for instance in self.instances:
            if instance.type.name not in unique_instance_types:
                unique_instance_types[instance.type.name] = instance.type
        return list(unique_instance_types.values())

    def _predict(self, _img: Any) -> List[PoseInstance]:
        return self.instances
