from copy import deepcopy
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import yaml
from py3r.pose.core.types import PoseInstanceType, PoseInstance, PosePoint
from torch import Tensor
from ultralytics import YOLO

from py3r.pose.core.model.pose_model import PoseModel


def load_instance_types(instance_types_folder: Path) -> List[dict]:
    instance_types_files = list(instance_types_folder.glob("*.yaml"))
    instance_types = []
    for instance_type_file in instance_types_files:
        with instance_type_file.open("r") as f:
            instance_type = yaml.safe_load(f)
        instance_types.append(instance_type)
    return instance_types


def expand_instance_types(instance_types: List[dict]) -> List[dict]:
    instance_types_map = {instance_type["name"]: instance_type for instance_type in instance_types}

    instance_types = deepcopy(instance_types)

    for instance_type in instance_types:
        parent = instance_type.get("parent")
        if parent is not None:
            parent_instance_type = instance_types_map[parent]
            instance_type["points"] = parent_instance_type["points"] + instance_type["points"]
            del instance_type["parent"]

    return instance_types


def parse_instance_types(instance_types: List[dict]) -> List[PoseInstanceType]:
    parsed_instance_types = []
    for instance_type in instance_types:
        name = instance_type["name"]
        points = instance_type["points"]
        skeleton = instance_type["skeleton"]

        point_names = [point["name"] for point in points]
        skeleton = [(point_names.index(skeleton_point[0]), point_names.index(skeleton_point[1])) for skeleton_point in skeleton]

        instance_type = PoseInstanceType(name, point_names, skeleton)
        parsed_instance_types.append(instance_type)
    return parsed_instance_types


class YoloPoseModel:
    def __init__(self, model: YOLO, instance_types: List[PoseInstanceType], output_mapping: dict, device: str = "cuda",
                 confidence_threshold: float = 0.1, iou_threshold: float = 0.1):
        self.model = model
        self.instance_types = instance_types
        self.output_mapping = output_mapping
        self.instance_type_map = {instance_type.name: instance_type for instance_type in instance_types}
        self.device = torch.device(device) if isinstance(device, str) else device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @classmethod
    def from_folder(cls, folder: Path, device: str = "cuda", confidence_threshold: float = 0.1, iou_threshold: float = 0.1):
        instance_types = cls.load_instance_types(folder)
        output_mapping = cls.load_output_mapping(folder)
        model_weights_file = folder / "weights.pt"
        if not model_weights_file.exists():
            model_weights_file = folder / "weights" / "best.pt"
        return cls(YOLO(str(model_weights_file)), instance_types, output_mapping, device, confidence_threshold, iou_threshold)

    @classmethod
    def load_instance_types(cls, model_folder: Path) -> List[PoseInstanceType]:
        instance_type_folder = model_folder / "meta" / "instance_types"
        instance_types = load_instance_types(instance_type_folder)
        instance_types = expand_instance_types(instance_types)
        instance_types = parse_instance_types(instance_types)
        return instance_types

    @classmethod
    def load_output_mapping(cls, model_folder: Path) -> dict:
        output_mapping_file = model_folder / "meta" / "output_mapping.csv"
        output_mapping = {}
        with output_mapping_file.open("r") as f:
            for line in f:
                line_split = line.strip().split(",")
                if len(line_split) == 3:
                    instance_name, point_name, output_index = line_split
                    output_mapping[instance_name, point_name] = int(output_index)
        return output_mapping

    def get_instance_types(self) -> List[PoseInstanceType]:
        return self.instance_types

    def _parse_result(self, result) -> List[PoseInstance]:
        class_name_map = result.names

        result = result.cpu().numpy()
        instance_confidences = list(result.boxes.conf)

        instances = []

        num_instances_per_type = {}

        for instance_index, instance_conf in enumerate(instance_confidences):
            instance_class = result.boxes.cls[instance_index]
            instance_type_name = class_name_map[round(instance_class)]

            # Ignore instances that are not in the instance type map (TODO: this is for testing, maybe remove later)
            if instance_type_name not in self.instance_type_map:
                continue

            instance_type = self.instance_type_map[instance_type_name]

            if instance_type_name not in num_instances_per_type:
                num_instances_per_type[instance_type_name] = 0
            num_instances_per_type[instance_type_name] += 1

            if result.boxes.id is not None:
                # When using yolo.track, we can use the id field to get the instance id
                # which should always refer to the same instance across frames
                instance_id = int(result.boxes.id[instance_index])
            else:
                # Otherwise, we use the per class instance count
                instance_id = num_instances_per_type[instance_type_name] - 1

            instance_id = f"{instance_type_name}_{instance_id}"

            box_coords = result.boxes.xyxy[instance_index]
            box_coords = (float(box_coords[0]), float(box_coords[1]), float(box_coords[2]), float(box_coords[3]))

            instance_points_xy = result.keypoints.xy[instance_index]

            point_confs = []
            point_coords = []
            for point_name in instance_type.point_names:
                output_index = self.output_mapping[instance_type_name, point_name]
                point_confs.append(result.keypoints.conf[instance_index][output_index])
                point_xy = instance_points_xy[output_index]
                point_coords.append(point_xy)

            points = [PosePoint(float(coords[0]), float(coords[1]), float(conf)) for conf, coords in zip(point_confs, point_coords)]

            instances.append(PoseInstance(instance_id, instance_type, box_coords, points, float(instance_conf)))

        return instances

    def predict(self, img: np.ndarray | Tensor) -> List[PoseInstance]:
        results = self.model(img, verbose=False, device=self.device, conf=self.confidence_threshold, iou=self.iou_threshold)
        return self._parse_result(results[0])

    def predict_batch(self, batch: Union[List[np.ndarray], Tensor]) -> List[List[PoseInstance]]:
        results = self.model(batch, verbose=False, device=self.device, conf=self.confidence_threshold, iou=self.iou_threshold)
        return [self._parse_result(result) for i, result in enumerate(results)]
