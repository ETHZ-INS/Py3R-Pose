from dataclasses import dataclass
from typing import List

from py3r.point_tracking.core.data.instance import Instance


@dataclass(slots=True)
class Frame:
    index: int
    width: int
    height: int
    instances: List[Instance]

    def as_dict(self):
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "instances": [instance.as_dict() for instance in self.instances]
        }
