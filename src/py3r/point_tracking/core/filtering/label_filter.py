from typing import List

from py3r.point_tracking.core.types.instance import PoseInstance


class LabelFilter:
    def filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        raise NotImplementedError

    def filter_all(self, frames: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        return [self.filter(frame) for frame in frames]
