from typing import Any, List

import numpy as np
from py3r.pose.core.model.pose_model import PoseModel
from py3r.pose.core.types import PoseInstanceType, PoseInstance

try:
    # noinspection PyPackageRequirements
    import torch
except ImportError:
    torch = None


class IntermittentPoseModel:
    def __init__(self, model: PoseModel, interval: int):
        self._model = model
        self._interval = interval
        self._frame_count = 0
        self._last_result = None

    @property
    def instance_types(self) -> List[PoseInstanceType]:
        return self._model.get_instance_types()

    def predict(self, img: np.ndarray | Any) -> List[PoseInstance]:
        if self._frame_count % self._interval == 0:
            self._last_result = self._model.predict(img)
        self._frame_count += 1
        return self._last_result

    def predict_batch(self, batch: Any) -> List[List[PoseInstance]]:
        if isinstance(batch, np.ndarray):
            batch = [batch[i] for i in range(batch.shape[0])]
        elif torch and isinstance(batch, torch.Tensor):
            batch = [batch[i:i+1] for i in range(batch.shape[0])]
        return [self.predict(img) for img in batch]
