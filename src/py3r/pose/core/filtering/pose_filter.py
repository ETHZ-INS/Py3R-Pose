from typing import List

from py3r.pose.core.types import HasPoses, Poses
from py3r.pose.core.types.instance import PoseInstance


class PoseFilter:
    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        raise NotImplementedError

    def _filter_all(self, instance_lists: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        return [self._filter(instances) for instances in instance_lists]

    def filter(self, pose_results: HasPoses | List[PoseInstance]) -> Poses:
        instances = pose_results if isinstance(pose_results, list) else pose_results.instances
        filtered_instances = self._filter(instances)
        return Poses(filtered_instances)

    def filter_all(self, pose_results_list: List[HasPoses] | List[List[PoseInstance]]) -> List[Poses]:
        if len(pose_results_list) == 0:
            return []

        if isinstance(pose_results_list[0], list):
            all_instances = pose_results_list
        else:
            all_instances = [pr.instances for pr in pose_results_list]

        filtered_instances_list = self._filter_all(all_instances)
        return [Poses(instances) for instances in filtered_instances_list]

