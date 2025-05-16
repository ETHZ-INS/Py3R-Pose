from typing import List

import numpy as np

from py3r.point_tracking.core.data.instance import Instance
from py3r.point_tracking.core.data.point import Point
from py3r.point_tracking.core.filtering.label_filter import LabelFilter


class MeanFilter(LabelFilter):
    def __init__(self, instance_whitelist: List[str] = None, replace_missing: bool = True):
        self.instance_whitelist = instance_whitelist
        self.replace_missing = replace_missing

    def filter(self, instances: List[Instance]) -> List[Instance]:
        raise NotImplementedError

    def filter_all(self, frames: List[List[Instance]]) -> List[List[Instance]]:
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

            mean_box = np.nanmean(boxes, axis=0)
            mean_conf = np.nanmean(confs)
            mean_points = np.nanmean(poses, axis=0)

            mean_box = (float(mean_box[0]), float(mean_box[1]), float(mean_box[2]), float(mean_box[3]))
            mean_points = [(float(mean_points[i][0]), float(mean_points[i][1]), float(mean_points[i][2])) for i in range(len(mean_points))]
            mean_points = [Point(mean_points[i][0], mean_points[i][1], mean_points[i][2]) for i in range(len(mean_points))]

            for frame in frames:
                found = False
                for instance in frame:
                    if instance.type.name != instance_type.name or instance.id != instance_id:
                        continue
                    instance.box = mean_box
                    instance.points = mean_points
                    instance.conf = mean_conf
                    found = True
                if not found and self.replace_missing:
                    frame.append(Instance(instance_id, instance_type, mean_box, mean_points, mean_conf))

        return frames
