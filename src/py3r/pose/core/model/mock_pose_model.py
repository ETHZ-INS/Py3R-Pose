from copy import deepcopy
from typing import List, Any, Union, Sequence

import numpy as np

from py3r.pose.core.model.pose_model import PoseModel
from py3r.pose.core.types import PoseInstance, PoseInstanceType


class MockPoseModel(PoseModel):
    def __init__(self, instances: List[PoseInstance], noise: float = 0.01):
        self.instances = instances
        self.noise = noise

    def get_instance_types(self) -> List[PoseInstanceType]:
        unique_instance_types = {}
        for instance in self.instances:
            if instance.type.name not in unique_instance_types:
                unique_instance_types[instance.type.name] = instance.type
        return list(unique_instance_types.values())

    def _predict(self, _img: Union[np.ndarray, Any]) -> List[PoseInstance]:
        # Add random noise to each point
        instances = [deepcopy(instance) for instance in self.instances]
        for instance in instances:
            for point in instance.points:
                point.x += np.random.normal(0, self.noise)
                point.y += np.random.normal(0, self.noise)
        return instances

    def _predict_batch(self, batch: Union[Sequence[np.ndarray], Any]) -> List[List[PoseInstance]]:
        return [self._predict(img) for img in batch]
