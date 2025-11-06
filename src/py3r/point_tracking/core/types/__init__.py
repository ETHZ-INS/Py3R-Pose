from dataclasses import dataclass, field, replace
from typing import List, Tuple, runtime_checkable, Protocol

import numpy as np

from py3r.point_tracking.core.types.instance_type import PoseInstanceType
from py3r.point_tracking.core.types.instance import PoseInstance
from py3r.point_tracking.core.types.point import PosePoint


@runtime_checkable
class HasImage(Protocol):
    @property
    def img(self) -> np.ndarray: ...

    @property
    def is_color(self) -> bool:
        return self.img.ndim == 3 and self.img.shape[2] == 3

@runtime_checkable
class HasImageMeta(Protocol):
    @property
    def size(self) -> Tuple[int, int]: ...  # width, height

@runtime_checkable
class HasFrameMeta(HasImageMeta, Protocol):
    @property
    def frame_index(self) -> int: ...
    @property
    def timestamp(self) -> float: ...

@runtime_checkable
class HasPoses(Protocol):
    @property
    def instances(self) -> List[PoseInstance]: ...

# Concrete dataclasses (single inheritance, no MI headaches)
@dataclass(frozen=True, slots=True)
class ImageMeta(HasImageMeta):  # implements HasImage & HasSize
    _size: Tuple[int, int]

    @classmethod
    def from_meta(cls, o: HasImageMeta) -> "ImageMeta":
        return cls(o.size)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

@dataclass(frozen=True, slots=True)
class FrameMeta(HasFrameMeta):  # implements HasImage & HasSize
    _size: Tuple[int, int]
    _frame_index: int = 0
    _timestamp: float = 0.0

    @classmethod
    def from_meta(cls, o: HasFrameMeta) -> "FrameMeta":
        return cls(o.size, o.frame_index, o.timestamp)

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp(self) -> float:
        return self._timestamp

@dataclass(frozen=True, slots=True)
class Image(HasImage, HasImageMeta):  # implements HasImage & HasSize
    _img: np.ndarray

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h

@dataclass(frozen=True, slots=True)
class VideoFrame(HasImage, HasFrameMeta):  # inherits only from Image concretely
    _img: np.ndarray
    _frame_index: int = 0
    _timestamp: float = 0.0

    @classmethod
    def from_parts(cls, img: HasImage | np.ndarray, meta: HasFrameMeta) -> "VideoFrame":
        return cls(
            _img=img if isinstance(img, np.ndarray) else img.img,
            _frame_index=meta.frame_index,
            _timestamp=meta.timestamp,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[HasImage | np.ndarray, HasFrameMeta]) -> "VideoFrame":
        return cls.from_parts(*pair)

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.img.shape[:2]
        return w, h

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp(self) -> float:
        return self._timestamp

@dataclass(frozen=True, slots=True)
class Poses(HasPoses):
    _instances: List[PoseInstance] = field(default_factory=list)

    @property
    def instances(self) -> List[PoseInstance]:
        return self._instances

@dataclass(frozen=True, slots=True)
class ImagePoses(HasPoses, HasImageMeta):
    _instances: List[PoseInstance] = field(default_factory=list)
    _size: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    @classmethod
    def from_parts(cls, pose_results: HasPoses, meta: HasImageMeta) -> "ImagePoses":
        return cls(
            _instances=pose_results.instances,
            _size=meta.size,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[HasPoses, HasImageMeta]) -> "ImagePoses":
        return cls.from_parts(*pair)

    def with_meta(self, meta: HasImageMeta) -> "ImagePoses":
        return replace(
            self,
            _size=meta.size,
        )

    def with_pose_results(self, pose_results: HasPoses) -> "ImagePoses":
        return replace(
            self,
            _instances=pose_results.instances,
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def instances(self) -> List[PoseInstance]:
        return self._instances

@dataclass(frozen=True, slots=True)
class VideoFramePoses(HasPoses, HasFrameMeta):
    _instances: List[PoseInstance] = field(default_factory=list)
    _size: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    _frame_index: int = 0
    _timestamp: float = 0.0

    @classmethod
    def from_parts(cls, poses: HasPoses, meta: HasFrameMeta) -> "VideoFramePoses":
        return cls(
            _instances=poses.instances,
            _size=meta.size,
            _frame_index=meta.frame_index,
            _timestamp=meta.timestamp,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[HasPoses, HasFrameMeta]) -> "VideoFramePoses":
        return cls.from_parts(*pair)

    def with_meta(self, meta: HasFrameMeta) -> "VideoFramePoses":
        return replace(
            self,
            _size=meta.size,
            _frame_index=meta.frame_index,
            _timestamp=meta.timestamp,
        )

    def with_pose_results(self, pose_results: HasPoses) -> "VideoFramePoses":
        return replace(
            self,
            _instances=pose_results.instances,
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def instances(self) -> List[PoseInstance]:
        return self._instances
