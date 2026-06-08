import difflib
from dataclasses import dataclass
from typing import List, Optional

from py3r.pose.core.filtering.abc.pose_filter import IPoseFilter
from py3r.pose.core.filtering.instance_scoped_filter import InstanceScopedFilter
from py3r.pose.core.filtering.sequential_filter import SequentialPoseFilter
from py3r.pose.cli.filter_chain_parser import parse_filter_chain
from py3r.pose.cli.filter_registry import FILTER_REGISTRY


@dataclass
class FilterStep:
    """
    A fully-built filter step ready for use in a single-stream pipeline.

    filter — an InstanceScopedFilter wrapping the inner filter class.

    Note: multi-stream routing (input source streams via <s:...>, output
    stream naming via @name) is not yet implemented. When added, FilterStep
    will gain input_streams and output_stream fields alongside a stream
    graph evaluator to replace the current linear SequentialPoseFilter usage.
    """
    filter: InstanceScopedFilter


def build_filter_chain(chain_str: str) -> List[FilterStep]:
    """
    Parse and build a filter chain from a filter chain string.

    Raises FilterSyntaxError on parse errors, ValueError on unknown filter
    names or invalid params.
    """
    specs = parse_filter_chain(chain_str)
    steps: List[FilterStep] = []

    for spec in specs:
        if spec.name not in FILTER_REGISTRY:
            close = difflib.get_close_matches(spec.name, FILTER_REGISTRY.keys(), n=1)
            hint = f", did you mean '{close[0]}'?" if close else ""
            raise ValueError(f"Unknown filter '{spec.name}'{hint}")

        descriptor = FILTER_REGISTRY[spec.name]
        resolved = descriptor.resolve_params(spec.params)
        inner = descriptor.build(resolved)

        subject_types: Optional[List[str]] = None
        if spec.subject_selector is not None:
            names = spec.subject_selector.instance_type_names
            subject_types = names if names else None

        context_types: Optional[List[str]] = (
            spec.context_selector.instance_type_names
            if spec.context_selector is not None
            else None
        )

        steps.append(FilterStep(InstanceScopedFilter(inner, subject_types, context_types)))

    return steps


def filters_from_chain(steps: List[FilterStep]) -> List[InstanceScopedFilter]:
    """Return the ordered list of filters from a built chain."""
    return [step.filter for step in steps]


def build_sequential_filter(chain_str: str) -> IPoseFilter:
    """
    Build a single composed pose filter object for the current single-stream implementation.

    The returned object can be passed directly to PredictionJob / RenderJob.
    """
    return SequentialPoseFilter(filters_from_chain(build_filter_chain(chain_str)))