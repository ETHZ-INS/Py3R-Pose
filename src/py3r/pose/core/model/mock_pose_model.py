from typing import List, Any, Union, Sequence

import numpy as np

from py3r.pose.core.model.pose_model import PoseModel
from py3r.pose.core.types import PoseInstance, PoseInstanceType


class MockPoseModel(PoseModel):
    def __init__(self, instances: List[PoseInstance]):
        self.instances = instances

    def get_instance_types(self) -> List[PoseInstanceType]:
        unique_instance_types = {}
        for instance in self.instances:
            if instance.type.name not in unique_instance_types:
                unique_instance_types[instance.type.name] = instance.type
        return list(unique_instance_types.values())

    def _predict(self, _img: Union[np.ndarray, Any]) -> List[PoseInstance]:
        return self.instances

    def _predict_batch(self, batch: Union[Sequence[np.ndarray], Any]) -> List[List[PoseInstance]]:
        return [self.instances for _ in batch]
