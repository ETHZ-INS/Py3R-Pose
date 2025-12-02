from typing import List, Union

from py3r.pose.core.types import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.filtering.pose_filter import PoseFilter


class InstanceTypePoseFilter(PoseFilter):
    def __init__(self, instance_types: List[Union[str, PoseInstanceType]], whitelist: bool = True):
        super().__init__()
        self.instance_types = [t if isinstance(t, str) else t.name for t in instance_types]
        self.whitelist = whitelist

    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        return [
            inst for inst in instances
            if (inst.type.name in self.instance_types) == self.whitelist
        ]
