from dataclasses import dataclass
from typing import List, Optional

from py3r.pose.core.types.instance_type import PoseInstanceType
from py3r.pose.core.types.point import PosePoint


@dataclass(slots=True)
class PoseInstance:
    id: str
    type: PoseInstanceType
    box: tuple[float, float, float, float]
    points: List[PosePoint]
    # conf is None for human-annotated instances
    conf: Optional[float] = None

    def as_dict(self):
        return {
            "id": self.id,
            "type": self.type.name,
            "box": {"x1": self.box[0], "y1": self.box[1], "x2": self.box[2], "y2": self.box[3], "conf": self.conf},
            "points": {point_name: point.as_dict() if point is not None else None for point_name, point in zip(self.type.point_names, self.points)}
        }
