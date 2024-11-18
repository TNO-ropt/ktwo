"""This module implements functions for use with the k2 plan plugin."""

from __future__ import annotations

from typing import Any

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from ropt.config.enopt import EnOptConfig


def fnc_everest2ropt(everest_config: dict[str, Any]) -> EnOptConfig:
    everest_config = EverestConfig.model_validate(everest_config)
    return EnOptConfig.model_validate(everest2ropt(everest_config))
