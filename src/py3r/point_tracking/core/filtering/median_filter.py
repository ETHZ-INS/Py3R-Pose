from typing import List

import numpy as np

from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.types.point import PosePoint
from py3r.point_tracking.core.filtering.label_filter import LabelFilter


class MedianFilter(LabelFilter):
    def __init__(self, instance_whitelist: List[str] = None, replace_missing: bool = True):
        self.instance_whitelist = instance_whitelist
        self.replace_missing = replace_missing

    def filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        raise NotImplementedError

    def filter_all(self, frames: List[List[PoseInstance]]) -> List[List[PoseInstance]]:
        unique_instances_set = set()
        unique_instances = []
        for frame in frames:
            for instance in frame:
                if self.instance_whitelist is not None and instance.type.name not in self.instance_whitelist:
                    continue
                if (instance.type.name, instance.id) in unique_instances_set:
                    continue
                unique_instances_set.add((instance.type.name, instance.id))
                unique_instances.append((instance.type, instance.id))

        for instance_type, instance_id in unique_instances:
            boxes = []
            poses = []
            confs = []

            for frame in frames:
                for instance in frame:
                    if instance.type != instance_type or instance.id != instance_id:
                        continue
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

            for frame in frames:
                found = False
                for instance in frame:
                    if instance.type.name != instance_type.name or instance.id != instance_id:
                        continue
                    instance.box = median_box
                    instance.points = median_points
                    instance.conf = median_conf
                    found = True
                if not found and self.replace_missing:
                    frame.append(PoseInstance(instance_id, instance_type, median_box, median_points, median_conf))

        return frames
