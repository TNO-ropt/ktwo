"""This module implements the default setvar step."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Union

from pydantic import BaseModel, ConfigDict
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import StepConfig
    from ropt.plan import Plan


class K2UpdateConfigStepWith(BaseModel):
    """Parameters used by the parse config step.

    The keys in the

    Attributes:
        input:   The input expression
        output:  The output variable
        updates: The new values
    """

    input: str
    output: str
    updates: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class K2UpdateConfigStep(PlanStep):
    """The default update config step."""

    def __init__(self, config: StepConfig, plan: Plan) -> None:
        """Initialize a update config step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = K2UpdateConfigStepWith.model_validate(config.with_)

    def run(self) -> None:
        """Run the update config step."""
        everest_config = _convert_lists_to_dicts(self._plan.eval(self._with.input))
        for update, value in self._with.updates.items():
            parsed_value = self._plan.eval(value)
            *keys, last_key = update.split("/")
            config = everest_config
            for key in keys:
                config = config.setdefault(int(key) if key.isdigit() else key, {})
            config[int(last_key) if last_key.isdigit() else last_key] = parsed_value
        self._plan[self._with.output] = _convert_dicts_to_lists(everest_config)


def _convert_lists_to_dicts(values: Dict[str, Any]) -> Dict[Union[int | str], Any]:
    result: Dict[Union[int | str], Any] = {}
    for key, value in values.items():
        if isinstance(value, list):
            result[key] = _convert_lists_to_dicts(dict(enumerate(value)))  # type: ignore[arg-type]
        elif isinstance(value, dict):
            result[key] = _convert_lists_to_dicts(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _convert_dicts_to_lists(values: Any) -> Dict[str, Any]:  # noqa: ANN401
    if not isinstance(values, dict):
        return values
    result: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict) and all(isinstance(idx, int) for idx in value):
            result[key] = [
                _convert_dicts_to_lists(value[idx])
                for idx in range(max(value.keys()) + 1)
                if idx in value
            ]
        else:
            result[key] = _convert_dicts_to_lists(value)
    return result
