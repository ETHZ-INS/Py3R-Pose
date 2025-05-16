from typing import List

from py3r.point_tracking.core.data.instance import Instance


class LabelFilter:
    def filter(self, instances: List[Instance]) -> List[Instance]:
        raise NotImplementedError

    def filter_all(self, frames: List[List[Instance]]) -> List[List[Instance]]:
        return [self.filter(frame) for frame in frames]
