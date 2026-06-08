from typing import List, Optional

from py3r.pose.core.types.instance import PoseInstance


class ArenaPoseFilter:
    """
    Filters out subject instances that don't overlap with at least one arena instance.

    arena_type:       type name used to identify arena instances within the context list
    min_intersection: minimum required overlap as a fraction of the subject instance's box area

    When used via InstanceScopedFilter with no explicit context selector, the full frame
    is passed as context automatically, so arena instances will be present without any
    extra configuration in the filter chain syntax.
    """

    def __init__(self, arena_type, min_intersection: float = 0.1):
        self.arena_types: List[str] = [arena_type] if isinstance(arena_type, str) else list(arena_type)
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

        return intersection > self.min_intersection

    def filter(self, instances: List[PoseInstance], context: Optional[List[PoseInstance]] = None) -> List[PoseInstance]:
        arena_instances = [i for i in (context or []) if i.type.name in self.arena_types]

        filtered_instances = []
        for instance in instances:
            for arena_instance in arena_instances:
                if self._bounding_box_overlap(instance.box, arena_instance.box):
                    filtered_instances.append(instance)
                    break

        return filtered_instances

    def filter_all(self, instance_lists: List[List[PoseInstance]], context_lists: Optional[List[List[PoseInstance]]] = None) -> List[List[PoseInstance]]:
        if context_lists:
            return [self.filter(instances, context) for instances, context in zip(instance_lists, context_lists)]
        return [self.filter(instances) for instances in instance_lists]
