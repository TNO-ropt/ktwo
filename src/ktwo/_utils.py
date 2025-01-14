"""This module implements utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.enums import ResultAxis

if TYPE_CHECKING:
    from everest.config import EverestConfig


def _get_names(
    everest_config: EverestConfig | None,
) -> dict[ResultAxis, tuple[str, ...] | None] | None:
    if everest_config is None:
        return None

    def _join(controls: tuple[str, str, int | tuple[str, str]]) -> str:
        if len(controls) == 3:  # noqa: PLR2004
            return f"{controls[0]}_{controls[1]}-{controls[2]}"
        return f"{controls[0]}_{controls[1]}"

    return {
        ResultAxis.VARIABLE: tuple(
            _join(control) for control in everest_config.control_name_tuples
        ),
        ResultAxis.OBJECTIVE: everest_config.objective_names,
        ResultAxis.NONLINEAR_CONSTRAINT: everest_config.constraint_names,
        ResultAxis.REALIZATION: everest_config.model.realizations,
    }
