"""This module implements functions for use with the k2 plan plugin."""

from __future__ import annotations

from typing import Any

from everest.config import EverestConfig
from ropt.results import Results

from ._utils import _get_names


def _results2dict(
    everest_config: dict[str, Any], results: Results, name: str
) -> dict[str | int, Any]:
    if not isinstance(results, Results):
        msg = "Cannot retrieve dict from results"
        raise TypeError(msg)
    names = _get_names(EverestConfig.model_validate(everest_config))
    field_name, sep, sub_field_name = name.partition(".")
    if sep != ".":
        msg = "Invalid field specification"
        raise RuntimeError(msg)
    field = getattr(results, field_name)
    return field.to_dict(sub_field_name, names=names)
