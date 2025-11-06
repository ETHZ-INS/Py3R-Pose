from dataclasses import dataclass


@dataclass(slots=True)
class PosePoint:
    x: float
    y: float
    # conf is None for human-annotated points
    conf: float | None = None

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "conf": self.conf}
