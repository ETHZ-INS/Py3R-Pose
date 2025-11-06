from typing import List

from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.filtering.pose_filter import PoseFilter


class SequentialPoseFilter(PoseFilter):
    def __init__(self, filters: List[PoseFilter]):
        self.filters = filters

    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        for label_filter in self.filters:
            instances = label_filter._filter(instances)
        return instances

    def _filter_all(self, instance_lists: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        for label_filter in self.filters:
            instance_lists = label_filter._filter_all(instance_lists)
        return instance_lists
