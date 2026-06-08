from typing import List

from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.types.instance import PoseInstance


class SequentialPoseFilter:
    def __init__(self, filters: List[IPoseFilter]):
        self.filters = filters

    def filter(self, instances: List[PoseInstance], context: List[PoseInstance] = []) -> List[PoseInstance]:
        for f in self.filters:
            instances = f.filter(instances)
        return instances

    def filter_all(self, instance_lists: List[List[PoseInstance]], context_lists: List[List[PoseInstance]] = []) -> List[List[PoseInstance]]:
        for f in self.filters:
            instance_lists = f.filter_all(instance_lists)
        return instance_lists
