from dataclasses import dataclass
from typing import Any, Dict, List

from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter


@dataclass
class FilterParamSpec:
    name: str
    type: type       # str | float | int | bool | list
    default: Any = None
    required: bool = False


class FilterDescriptor:
    """
    Describes a named filter type: its accepted parameters and how to instantiate
    the inner filter class. Registered in FILTER_REGISTRY by filter_name.
    """

    def __init__(self, filter_name: str, params: List[FilterParamSpec]):
        self.filter_name = filter_name
        self.params = params

    def resolve_params(self, raw_params: list) -> dict:
        """
        Resolve a list of (name_or_None, value) tuples into a fully-named dict,
        applying positional mapping and defaults. Raises ValueError on unknown,
        duplicate, missing-required, or wrongly-typed params.
        """
        resolved: Dict[str, Any] = {}
        positional_idx = 0

        for param_name, value in raw_params:
            if param_name is not None:
                spec = next((p for p in self.params if p.name == param_name), None)
                if spec is None:
                    known = ", ".join(p.name for p in self.params)
                    raise ValueError(
                        f"Filter '{self.filter_name}': unknown param '{param_name}'. "
                        f"Known params: {known}"
                    )
                if param_name in resolved:
                    raise ValueError(f"Filter '{self.filter_name}': duplicate param '{param_name}'")
                resolved[param_name] = self._coerce(value, spec)
            else:
                # advance past already-filled named params
                while positional_idx < len(self.params) and self.params[positional_idx].name in resolved:
                    positional_idx += 1
                if positional_idx >= len(self.params):
                    raise ValueError(f"Filter '{self.filter_name}': too many positional params")
                spec = self.params[positional_idx]
                resolved[spec.name] = self._coerce(value, spec)
                positional_idx += 1

        # apply defaults / check required
        for spec in self.params:
            if spec.name not in resolved:
                if spec.required:
                    raise ValueError(f"Filter '{self.filter_name}': required param '{spec.name}' is missing")
                resolved[spec.name] = spec.default

        return resolved

    def _coerce(self, value: Any, spec: FilterParamSpec) -> Any:
        if spec.type is list:
            if isinstance(value, str):
                return [value]
            if not isinstance(value, list):
                raise ValueError(f"Filter '{self.filter_name}': param '{spec.name}' must be a list or a single string")
            return value
        if spec.type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Filter '{self.filter_name}': param '{spec.name}' must be true or false")
            return value
        if spec.type is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            raise ValueError(f"Filter '{self.filter_name}': param '{spec.name}' must be a number")
        if spec.type is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            raise ValueError(f"Filter '{self.filter_name}': param '{spec.name}' must be an integer")
        if spec.type is str:
            if not isinstance(value, str):
                raise ValueError(f"Filter '{self.filter_name}': param '{spec.name}' must be a string")
            return value
        return value

    def build(self, resolved_params: dict) -> IPoseFilter:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTER_REGISTRY: Dict[str, FilterDescriptor] = {}


def register_filter(descriptor: FilterDescriptor) -> None:
    FILTER_REGISTRY[descriptor.filter_name] = descriptor


# ---------------------------------------------------------------------------
# Built-in descriptors
# ---------------------------------------------------------------------------

class _ConfidenceDescriptor(FilterDescriptor):
    def __init__(self):
        super().__init__("confidence", [
            FilterParamSpec("instance_threshold", float, default=0.5),
            FilterParamSpec("point_threshold", float, default=0.5),
        ])

    def build(self, p: dict) -> IPoseFilter:
        from py3r.pose.core.filtering.confidence_filter import ConfidencePoseFilter
        return ConfidencePoseFilter(
            instance_confidence_threshold=p["instance_threshold"],
            point_confidence_threshold=p["point_threshold"],
        )


class _ArenaDescriptor(FilterDescriptor):
    def __init__(self):
        super().__init__("arena", [
            FilterParamSpec("arena_type", list, required=True),
            FilterParamSpec("min_intersection", float, default=0.1),
        ])

    def build(self, p: dict) -> IPoseFilter:
        from py3r.pose.core.filtering.arena_filter import ArenaPoseFilter
        return ArenaPoseFilter(arena_type=p["arena_type"], min_intersection=p["min_intersection"])


class _InstanceTypeDescriptor(FilterDescriptor):
    def __init__(self):
        super().__init__("instance_type", [
            FilterParamSpec("types", list, required=True),
            FilterParamSpec("whitelist", bool, default=True),
        ])

    def build(self, p: dict) -> IPoseFilter:
        from py3r.pose.core.filtering.instance_type_filter import InstanceTypePoseFilter
        return InstanceTypePoseFilter(instance_types=p["types"], whitelist=p["whitelist"])


class _MedianDescriptor(FilterDescriptor):
    def __init__(self):
        super().__init__("median", [
            FilterParamSpec("replace_missing", bool, default=True),
        ])

    def build(self, p: dict) -> IPoseFilter:
        from py3r.pose.core.filtering.median_filter import MedianPoseFilter
        return MedianPoseFilter(replace_missing=p["replace_missing"])


register_filter(_ConfidenceDescriptor())
register_filter(_ArenaDescriptor())
register_filter(_InstanceTypeDescriptor())
register_filter(_MedianDescriptor())
