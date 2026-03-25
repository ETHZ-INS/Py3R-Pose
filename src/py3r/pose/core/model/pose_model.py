from abc import ABC, abstractmethod
from typing import List, Any

from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.instance_type import PoseInstanceType


class PoseModel(ABC):
    @property
    @abstractmethod
    def instance_types(self) -> List[PoseInstanceType]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, img: Any) -> List[PoseInstance]:
        """
        Predict poses for a single image.
        :param img: Input image (e.g., numpy array or tensor)
        :return: List of Instance objects representing detected poses
        """
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, batch: Any) -> List[List[PoseInstance]]:
        """
        Predict poses for a batch of images.
        :param batch: List of input images (e.g., numpy arrays or tensors).
        May accept batched numpy arrays/tensors depending on implementation.
        :return: List of Instance objects for each image in the batch
        """
        raise NotImplementedError
