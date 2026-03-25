from copy import deepcopy
from typing import List, Callable

import numpy as np

from py3r.pose.core.types import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.point import PosePoint


class MedianPoseFilter:
    def __init__(self, instance_type_filter: Callable[[PoseInstanceType], bool] = None, replace_missing: bool = True):
        self.instance_type_filter = instance_type_filter
        self.replace_missing = replace_missing

    @staticmethod
    def _calculate_median_instance(instances: List[PoseInstance]) -> PoseInstance:
        boxes = []
        poses = []
        confs = []

        for instance in instances:
            boxes.append(instance.box if instance.box is not None else [np.nan, np.nan, np.nan, np.nan])
            confs.append(instance.conf if instance.conf is not None else np.nan)

            pose = [(p.x, p.y, p.conf) if p is not None else (np.nan, np.nan, np.nan) for p in instance.points]
            pose = np.array(pose)
            poses.append(pose)

        boxes = np.array(boxes)
        confs = np.array(confs)
        poses = np.array(poses)

        median_box = np.nanmedian(boxes, axis=0)
        median_conf = float(np.nanmedian(confs))
        median_points = np.nanmedian(poses, axis=0)

        median_box = (float(median_box[0]), float(median_box[1]), float(median_box[2]), float(median_box[3]))
        median_points = [(float(median_points[i][0]), float(median_points[i][1]), float(median_points[i][2])) for i in range(len(median_points))]
        median_points = [PosePoint(median_points[i][0], median_points[i][1], median_points[i][2]) for i in range(len(median_points))]

        median_instance = PoseInstance(instances[0].id, instances[0].type, median_box, median_points, median_conf)
        return median_instance

    def filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        raise NotImplementedError

    def filter_all(self, instance_lists: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        unique_instances_set = set()
        unique_instances = []
        for frame in instance_lists:
            for instance in frame:
                if self.instance_type_filter is not None and not self.instance_type_filter(instance.type):
                    continue
                if (instance.type.name, instance.id) in unique_instances_set:
                    continue
                unique_instances_set.add((instance.type.name, instance.id))
                unique_instances.append((instance.type, instance.id))

        filtered_instance_lists = [
            [instance for instance in instances if (instance.type.name, instance.id) not in unique_instances_set]
            for instances in instance_lists
        ]

        for instance_type, instance_id in unique_instances:
            median_instance = self._calculate_median_instance([
                instance
                for frame in instance_lists
                for instance in frame
                if instance.type.name == instance_type and instance.id == instance_id
            ])

            for instances, filtered_instances in zip(instance_lists, filtered_instance_lists):
                if self.replace_missing or any(instance.type.name == instance_type.name and instance.id == instance_id for instance in instances):
                    filtered_instances.append(deepcopy(median_instance))

        return instance_lists
