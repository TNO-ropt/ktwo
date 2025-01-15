"""This module implements utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ropt.enums import ResultAxis

if TYPE_CHECKING:
    from everest.config import EverestConfig


def _get_names(
    everest_config: EverestConfig | None,
) -> dict[str, Sequence[str] | None] | None:
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


# ruff: noqa: ERA001,TD002,TD003,FIX002

# TODO: Change to this when formatted_control_names is available:
#
# def _get_names(
#     everest_config: EverestConfig | None,
# ) -> dict[str, Sequence[str] | None] | None:
#     return (
#         None
#         if everest_config is None
#         else {
#             ResultAxis.VARIABLE: everest_config.formatted_control_names,
#             ResultAxis.OBJECTIVE: everest_config.objective_names,
#             ResultAxis.NONLINEAR_CONSTRAINT: everest_config.constraint_names,
#             ResultAxis.REALIZATION: everest_config.model.realizations,
#         }
#     )
