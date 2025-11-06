from typing import List, Any, Iterable

from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.instance_type import PoseInstanceType
from py3r.pose.core.model.pose_model import PoseModel


class CompositePoseModel(PoseModel):
    """
    A PoseModel that combines multiple PoseModels.
    For each frame, the results from all models are combined into a single list of instances.
    """
    def __init__(self, models: List[PoseModel]):
        self._models = models
        # All models must use different class names so instance ids are unique
        # TODO: Is there a way to ensure this?

    def get_instance_types(self) -> List[PoseInstanceType]:
        instance_types = []
        for model in self._models:
            instance_types.extend(model.get_instance_types())
        return instance_types

    def _predict(self, img: Any) -> List[PoseInstance]:
        instances = []
        for model in self._models:
            instances.extend(model._predict(img))
        return instances

    def _predict_batch(self, batch: Any) -> List[List[PoseInstance]]:
        instances = [[] for _ in batch]

        for model in self._models:
            model_instances = model._predict_batch(batch)
            for i, instance_list in enumerate(model_instances):
                instances[i].extend(instance_list)

        return instances
