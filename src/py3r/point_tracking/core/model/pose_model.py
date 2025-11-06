from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Any

from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.types.instance_type import PoseInstanceType


class PoseModel(ABC):
    def get_instance_types(self) -> List[PoseInstanceType]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, img: Any) -> List[PoseInstance]:
        """
        Predict poses for a single image.
        :param img: Input image (e.g., numpy array or tensor)
        :return: List of Instance objects representing detected poses
        """
        raise NotImplementedError

    def predict_batch(self, batch: Iterable[Any]) -> List[List[PoseInstance]]:
        """
        Predict poses for a batch of images.
        :param batch: List of input images (e.g., numpy arrays or tensors).
        May accept batched numpy arrays/tensors depending on implementation.
        :return: List of Instance objects for each image in the batch
        """
        if not isinstance(batch, Iterable):
            raise ValueError("Batch must be an iterable of images")
        return [self.predict(img) for img in batch]
