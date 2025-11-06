from typing import List

from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.filtering.pose_filter import PoseFilter


class ConfidencePoseFilter(PoseFilter):
    def __init__(self, instance_confidence_threshold: float = 0.5, point_confidence_threshold: float = 0.5):
        self.instance_confidence_threshold = instance_confidence_threshold
        self.point_confidence_threshold = point_confidence_threshold

    def filter_points(self, instance: PoseInstance) -> PoseInstance:
        instance.points = [
            point if point is not None and point.conf >= self.point_confidence_threshold and not(point.x < 0.01 and point.y < 0.01) else None
            for point in instance.points
        ]
        return instance

    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        instances = [self.filter_points(instance) for instance in instances]
        # Filter out instances with low confidence and no points
        instances = [
            instance for instance in instances
            if instance.conf >= self.instance_confidence_threshold
        ]
        return instances
