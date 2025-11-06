from typing import List

from py3r.point_tracking.core.types import PoseInstanceType
from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.filtering.pose_filter import PoseFilter


class InstanceTypePoseFilter(PoseFilter):
    def __init__(self, instance_types: List[str | PoseInstanceType], whitelist: bool = True):
        super().__init__()
        self.instance_types = [itype if isinstance(itype, str) else itype.name for itype in instance_types]
        self.whitelist = whitelist

    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        return [
            inst for inst in instances
            if (inst.type.name in self.instance_types) == self.whitelist
        ]
