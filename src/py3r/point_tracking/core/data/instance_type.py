from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(slots=True)
class InstanceType:
    name: str
    point_names: List[str]
    skeleton: List[Tuple[int, int]] = field(default_factory=list)
