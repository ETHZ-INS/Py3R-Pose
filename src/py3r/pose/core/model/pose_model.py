from abc import ABC, abstractmethod
from typing import List, Any, Sequence, Union

import numpy as np

from py3r.media.types import HasImage
from py3r.pose.core.types import Poses
from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.instance_type import PoseInstanceType


class PoseModel(ABC):
    @abstractmethod
    def get_instance_types(self) -> List[PoseInstanceType]:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, img: Union[np.ndarray, Any]) -> List[PoseInstance]:
        """
        Predict poses for a single image.
        :param img: Input image (e.g., numpy array or tensor)
        :return: List of Instance objects representing detected poses
        """
        raise NotImplementedError

    @abstractmethod
    def _predict_batch(self, batch: Union[Sequence[np.ndarray], Any]) -> List[List[PoseInstance]]:
        """
        Predict poses for a batch of images.
        :param batch: List of input images (e.g., numpy arrays or tensors).
        May accept batched numpy arrays/tensors depending on implementation.
        :return: List of Instance objects for each image in the batch
        """
        raise NotImplementedError

    def predict(self, img: Any) -> Poses:
        if isinstance(img, HasImage):
            img = img.img
        return Poses(self._predict(img))

    def predict_batch(self, batch: Union[Sequence[HasImage], Sequence[np.ndarray], Any]) -> List[Poses]:
        if isinstance(batch, Sequence):
            if len(batch) == 0:
                return []
            elif isinstance(batch[0], HasImage):
                imgs = [item.img for item in batch]
                instance_lists = self._predict_batch(imgs)
            else:
                instance_lists = self._predict_batch(batch)
        else:
            instance_lists = self._predict_batch(batch)
        return [Poses(instances) for instances in instance_lists]
