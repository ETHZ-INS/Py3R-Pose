from typing import List, Sequence

import numpy as np
from py3r.pose.core.model.pose_model import PoseModel
from py3r.pose.core.types import PoseInstance, PoseInstanceType

from py3r.pose.yolo.model.yolo_batch_stager import YOLOBatchStager
from py3r.pose.yolo.preprocessing.letterbox import Letterbox


class StagedYoloPoseModel:
    def __init__(self, base_model: PoseModel, max_batch: int = 16, image_size: int = 640, stride: int = 32,
                 input_channels: int = 3, device: str = "cuda"):
        self.base_model = base_model
        self.stager = YOLOBatchStager(b=max_batch, imgsz=image_size, in_channels=input_channels, device=device)
        self.preprocessor = Letterbox(imgsz=image_size, stride=stride)

    def get_instance_types(self) -> List[PoseInstanceType]:
        return self.base_model.instance_types

    def predict(self, img: np.ndarray) -> List[PoseInstance]:
        return self.predict_batch([img])[0]

    def predict_batch(self, batch: Sequence[np.ndarray]) -> List[List[PoseInstance]]:
        preprocessed_batch, metas = self.preprocessor.forward_batch(batch)
        staged_batch = self.stager.upload(preprocessed_batch)
        pose_lists = self.base_model.predict_batch(staged_batch)
        pose_lists = [self.preprocessor.invert_poses(poses, meta) for poses, meta in zip(pose_lists, metas)]
        return pose_lists
