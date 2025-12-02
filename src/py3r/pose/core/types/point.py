from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PosePoint:
    x: float
    y: float
    # conf is None for human-annotated points
    conf: Optional[float] = None

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "conf": self.conf}
