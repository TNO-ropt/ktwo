# ruff:  noqa

from __future__ import annotations

from typing import Any

from ropt.config.enopt import EnOptConfig
from ropt.results import FunctionResults


def _fnc_objectives(config: EnOptConfig, data: Any) -> dict[str, Any]:
    if isinstance(data, FunctionResults):
        assert config.objective_functions.names is not None
        return {
            name: data.evaluations.objectives[:, idx]
            for idx, name in enumerate(config.objective_functions.names)
        }
    return {}


functions = {"get_objectives": _fnc_objectives}
