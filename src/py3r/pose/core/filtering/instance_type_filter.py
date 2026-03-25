from typing import List, Union

from py3r.pose.core.types import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance


class InstanceTypePoseFilter:
    def __init__(self, instance_types: List[Union[str, PoseInstanceType]], whitelist: bool = True):
        super().__init__()
        self.instance_types = [t if isinstance(t, str) else t.name for t in instance_types]
        self.whitelist = whitelist

    def filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        return [
            inst for inst in instances
            if (inst.type.name in self.instance_types) == self.whitelist
        ]

    def filter_all(self, instance_lists: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        return [self.filter(instances) for instances in instance_lists]
