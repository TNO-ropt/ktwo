"""This module implements functions for use with the k2 plan plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from ropt.results import ResultField

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig


def _everest2ropt(everest_config: dict[str, Any]) -> EnOptConfig:
    everest_config = EverestConfig.model_validate(everest_config)
    return everest2ropt(everest_config)


def _results2dict(
    config: EnOptConfig, field: ResultField, name: str
) -> dict[str | int, Any]:
    if not isinstance(field, ResultField):
        msg = "Cannot retrieve dict from results"
        raise TypeError(msg)
    return field.to_dict(config, name)
