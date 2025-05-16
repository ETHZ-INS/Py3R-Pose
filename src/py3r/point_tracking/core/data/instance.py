from dataclasses import dataclass
from typing import List

from py3r.point_tracking.core.data.instance_type import InstanceType
from py3r.point_tracking.core.data.point import Point


@dataclass(slots=True)
class Instance:
    id: str
    type: InstanceType
    box: tuple[float, float, float, float]
    points: List[Point]
    # conf is None for human-annotated instances
    conf: float | None  = None

    def as_dict(self):
        return {
            "id": self.id,
            "type": self.type.name,
            "box": {"x1": self.box[0], "y1": self.box[1], "x2": self.box[2], "y2": self.box[3], "conf": self.conf},
            "points": {point_name: point.as_dict() if point is not None else None for point_name, point in zip(self.type.point_names, self.points)}
        }
