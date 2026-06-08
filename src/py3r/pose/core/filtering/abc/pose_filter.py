from typing import List, Protocol

from py3r.pose.core.types.instance import PoseInstance


class IPoseFilter(Protocol):
    def filter(self, instances: List[PoseInstance], context: List[PoseInstance] = []) -> List[PoseInstance]: ...
    def filter_all(self, instance_lists: List[List[PoseInstance]], context_lists: List[List[PoseInstance]] = []) -> List[List[PoseInstance]]: ...
