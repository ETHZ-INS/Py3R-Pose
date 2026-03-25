from typing import List, Protocol

from py3r.pose.core.types.instance import PoseInstance


class IPoseFilter(Protocol):
    def filter(self, pose_results: List[PoseInstance]) -> List[PoseInstance]: ...
    def filter_all(self, pose_results_list: List[List[PoseInstance]]) -> List[List[PoseInstance]]: ...
