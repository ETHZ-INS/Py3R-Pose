from dataclasses import dataclass, field, replace
from typing import List, Tuple, runtime_checkable, Protocol

from py3r.media.types import HasImageMeta, HasFrameMeta

from py3r.pose.core.types.instance_type import PoseInstanceType
from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.point import PosePoint


@runtime_checkable
class HasPoses(Protocol):
    @property
    def instances(self) -> List[PoseInstance]: ...

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
