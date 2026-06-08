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
class ImagePoses(HasImageMeta):
    instances: List[PoseInstance] = field(default_factory=list)
    size: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    @classmethod
    def from_parts(cls, poses: List[PoseInstance], meta: HasImageMeta) -> "ImagePoses":
        return cls(
            instances=poses,
            size=meta.size,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[List[PoseInstance], HasImageMeta]) -> "ImagePoses":
        return cls.from_parts(*pair)

    def with_meta(self, meta: HasImageMeta) -> "ImagePoses":
        return replace(
            self,
            size=meta.size,
        )

    def with_poses(self, poses: List[PoseInstance]) -> "ImagePoses":
        return replace(
            self,
            instances=poses,
        )

@dataclass(frozen=True, slots=True)
class VideoFramePoses(HasFrameMeta):
    instances: List[PoseInstance] = field(default_factory=list)
    size: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    frame_index: int = 0
    timestamp: float = 0.0

    @classmethod
    def from_parts(cls, poses: List[PoseInstance], meta: HasFrameMeta) -> "VideoFramePoses":
        return cls(
            instances=poses,
            size=meta.size,
            frame_index=meta.frame_index,
            timestamp=meta.timestamp,
        )

    @classmethod
    def from_pair(cls, pair: Tuple[List[PoseInstance], HasFrameMeta]) -> "VideoFramePoses":
        return cls.from_parts(*pair)

    def with_meta(self, meta: HasFrameMeta) -> "VideoFramePoses":
        return replace(
            self,
            size=meta.size,
            frame_index=meta.frame_index,
            timestamp=meta.timestamp,
        )

    def with_poses(self, poses: List[PoseInstance]) -> "VideoFramePoses":
        return replace(
            self,
            instances=poses,
        )
