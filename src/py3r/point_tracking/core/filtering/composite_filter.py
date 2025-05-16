from typing import List

from py3r.point_tracking.core.data.instance import Instance
from py3r.point_tracking.core.filtering.label_filter import LabelFilter


class SequentialLabelFilter(LabelFilter):
    def __init__(self, filters: List[LabelFilter]):
        self.filters = filters

    def filter(self, instances: List[Instance]) -> List[Instance]:
        for label_filter in self.filters:
            instances = label_filter.filter(instances)
        return instances

    def filter_all(self, frames: List[List[Instance]]) -> List[List[Instance]]:
        for label_filter in self.filters:
            frames = label_filter.filter_all(frames)
        return frames
