"""This module implements the default setvar step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from pydantic import BaseModel, ConfigDict
from ropt.plan import RunStep

if TYPE_CHECKING:
    from ropt.config.plan import RunStepConfig
    from ropt.plan import Plan


class K2ParseEverestConfigStepWith(BaseModel):
    """Parameters used by the parse config step.

    Attributes:
        input:  The input expression
        output: The output variable
    """

    input: str
    output: str

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class K2ParseEverestConfigStep(RunStep):
    """The default parse config step."""

    def __init__(self, config: RunStepConfig, plan: Plan) -> None:
        """Initialize a default parse config step.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)

        self._with = K2ParseEverestConfigStepWith.model_validate(config.with_)

    def run(self) -> None:
        """Run the parse config step."""
        everest_config = EverestConfig.model_validate(self._plan.eval(self._with.input))
        self._plan[self._with.output] = everest2ropt(everest_config)
