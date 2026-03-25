from typing import List, Callable

from py3r.pose.core.types import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance


class ArenaPoseFilter:
    def __init__(self, arena_type_filter: Callable[[PoseInstanceType], bool], min_intersection: float = 0.1):
        # Filter out instances that don't overlap with at least one arena instance
        # Used to filter out instances outside the arena, which are probably false positives
        # arena_type_name: Name of the arena type
        # min_intersection: Minimum intersection required (as fraction of instance box area)

        self.arena_type_filter = arena_type_filter
        self.min_intersection = min_intersection

    def _bounding_box_overlap(self, instance_box, arena_box):
        x1, y1, x2, y2 = instance_box
        x3, y3, x4, y4 = arena_box

        if x1 > x4 or x3 > x2:
            return False
        if y1 > y4 or y3 > y2:
            return False

        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        intersection = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        instance_area = (x2 - x1) * (y2 - y1)

        intersection /= instance_area

        # Intersection (as fraction of instance box area) is greater than min_overlap
        if intersection > self.min_intersection:
            return True

        return False

    def filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        arena_instances = [
            instance for instance in instances
            if self.arena_type_filter(instance.type)
        ]

        # Filter out instances that don't overlap with at least one arena instance
        filtered_instances = []
        for instance in instances:
            for arena_instance in arena_instances:
                if self._bounding_box_overlap(instance.box, arena_instance.box):
                    filtered_instances.append(instance)
                    break

        return filtered_instances

    def filter_all(self, instance_lists: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        return [self.filter(instances) for instances in instance_lists]
