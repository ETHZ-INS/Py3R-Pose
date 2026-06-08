from typing import List

from py3r.pose.core.types import PosePoint
from py3r.pose.core.types.instance import PoseInstance


class ConfidencePoseFilter:
    def __init__(self, instance_confidence_threshold: float = 0.5, point_confidence_threshold: float = 0.5):
        self.instance_confidence_threshold = instance_confidence_threshold
        self.point_confidence_threshold = point_confidence_threshold

    def _filter_points(self, instance: PoseInstance) -> PoseInstance:
        instance = PoseInstance(instance.id, instance.type, instance.box, instance.points, instance.conf)
        instance.points = [
            # TODO: conf is currently allowed to be None, meaning human annotated or full confidence, that's a bit weird
            point if point is not None and point.conf >= self.point_confidence_threshold and not(point.x < 0.01 and point.y < 0.01) else None
            for point in instance.points
        ]
        return instance

    def filter(self, instances: List[PoseInstance], context: List[PoseInstance] = []) -> List[PoseInstance]:
        instances = [self._filter_points(instance) for instance in instances]
        # Filter out instances with low confidence and no points
        instances = [
            instance for instance in instances
            if instance.conf >= self.instance_confidence_threshold
        ]
        return instances

    def filter_all(self, instance_lists: List[List[PoseInstance]], context_lists: List[List[PoseInstance]] = []) -> List[List[PoseInstance]]:
        return [self.filter(instances) for instances in instance_lists]
