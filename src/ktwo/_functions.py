"""This module implements functions for use with the k2 plan plugin."""

from __future__ import annotations

from typing import Any, Dict

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt


def fnc_everest2ropt(everest_config: Dict[str, Any]) -> Dict[str, Any]:
    everest_config = EverestConfig.model_validate(everest_config)
    return everest2ropt(everest_config)
