"""This module implements functions for use with the k2 plan plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from ropt.results import Results

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig


def _everest2ropt(everest_config: dict[str, Any]) -> EnOptConfig:
    everest_config = EverestConfig.model_validate(everest_config)
    return everest2ropt(everest_config)


def _results2dict(results: Results, name: str) -> dict[str | int, Any]:
    if not isinstance(results, Results):
        msg = "Cannot retrieve dict from results"
        raise TypeError(msg)
    field_name, sep, sub_field_name = name.partition(".")
    if sep != ".":
        msg = "Invalid field specification"
        raise RuntimeError(msg)
    field = getattr(results, field_name)
    return field.to_dict(results.config, sub_field_name)
