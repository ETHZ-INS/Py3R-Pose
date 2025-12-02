import itertools
from typing import List, Union

from py3r.pose.core.types.instance_type import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.filtering.pose_filter import PoseFilter


class Track:
    def __init__(self, track_id: str):
        self.track_id = track_id
        self.start_frame = None
        self.last_pose = None

    def update(self, pose: PoseInstance):
        self.last_pose = pose


MISSING_POINT_PENALTY = 0.1


def pose_distance(pose1: PoseInstance, pose2: PoseInstance) -> float:
    instance_1_diag = ((pose1.box[2] - pose1.box[0]) ** 2 + (pose1.box[3] - pose1.box[1]) ** 2) ** 0.5
    instance_2_diag = ((pose2.box[2] - pose2.box[0]) ** 2 + (pose2.box[3] - pose2.box[1]) ** 2) ** 0.5
    scale = (instance_1_diag + instance_2_diag) / 2

    dist = 0
    missing_point_penalty = 0
    for p1, p2 in zip(pose1.points, pose2.points):
        if p1 is None and p2 is None:
            continue
        if p1 is None or p2 is None:
            missing_point_penalty += MISSING_POINT_PENALTY
            continue
        point_dist = (((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 2) / scale
        dist += point_dist
    dist += missing_point_penalty
    return dist / len(pose1.points)


pose_dists = []
time_dists = []


class FixedInstancesTracker(PoseFilter):
    """
    This tracker can be used when you know exactly how many instances of each type there are in the video.
    It has the advantage that it smartly filters out false positives by discarding instances
    that did not appear in the previous frame.
    """
    def __init__(self, instances: List[Union[PoseInstanceType, str]]):
        self.instance_types = [instance_type.name if isinstance(instance_type, PoseInstanceType) else instance_type for instance_type in instances]

        self.num_instances_per_type = {instance_type: 0 for instance_type in instances}
        for instance_type in instances:
            self.num_instances_per_type[instance_type] += 1

        self.tracks = {}
        for instance_type in instances:
            if instance_type not in self.tracks:
                self.tracks[instance_type] = []
            track_id = instance_type + f"_{len(self.tracks[instance_type])}"
            self.tracks[instance_type].append(Track(track_id))

    def _filter(self, instances: List[PoseInstance]) -> List[PoseInstance]:
        instances = [instance for instance in instances if any([point is not None for point in instance.points])]
        tracked_instances = []

        for instance_type_name, num_instances in self.num_instances_per_type.items():
            instance_type_tracks = self.tracks[instance_type_name]
            instance_type_instances = [instance for instance in instances if instance.type.name == instance_type_name]

            if len(instance_type_tracks) == len(instance_type_instances) == 1:
                instance_type_tracks[0].update(instance_type_instances[0])
                instance_type_instances[0].id = instance_type_tracks[0].track_id
                tracked_instances.append(instance_type_instances[0])
                continue

            dist_cache = {}
            for instance in instance_type_instances:
                for track in instance_type_tracks:
                    if track.last_pose is None:
                        dist = 0.
                    else:
                        dist = pose_distance(track.last_pose, instance)
                    dist_cache[(track.track_id, instance.id)] = dist

            min_total_dist = 1e12
            best_mapping = None

            # The mapping maps each instance to a track or None if there are more instances than tracks
            # and each track to an instance or None if there are more tracks than instances
            # no instance or track can be mapped to more than one other instance or track

            # Iterate over all possible mappings
            padded_tracks = instance_type_tracks + [None] * max(len(instance_type_instances) - len(instance_type_tracks), 0)
            padded_instances = instance_type_instances + [None] * max(len(instance_type_tracks) - len(instance_type_instances), 0)

            for track_list in itertools.permutations(padded_tracks):
                total_dist = 0
                for track, instance in zip(track_list, padded_instances):
                    if track is None or instance is None:
                        continue
                    total_dist += dist_cache[(track.track_id, instance.id)]
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_mapping = (track_list, padded_instances)

            if best_mapping is None:
                #tracked_instances.extend(instance_type_instances)
                continue

            for track, instance in zip(best_mapping[0], best_mapping[1]):
                if track is not None and instance is not None:
                    track.update(instance)
                    instance.id = track.track_id
                    tracked_instances.append(instance)
        return tracked_instances
