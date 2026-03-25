from typing import List, Any, Protocol

from py3r.pose.core.types.instance import PoseInstance
from py3r.pose.core.types.instance_type import PoseInstanceType


class IPoseModel(Protocol):
    @property
    def instance_types(self) -> List[PoseInstanceType]: ...

    def predict(self, img: Any) -> List[PoseInstance]: ...
    def predict_batch(self, batch: Any) -> List[List[PoseInstance]]: ...
