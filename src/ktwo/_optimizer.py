"""This module implements the K2 optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from ropt.plugins.plan.optimizer import DefaultOptimizerStep

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Event, Plan


class K2OptimizerStep(DefaultOptimizerStep):
    """The K2 optimizer step."""

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a K2 optimizer step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._everest_config: EverestConfig

    def parse_config(self, config: str) -> EnOptConfig:
        """Parse the configuration of the step.

        Returns:
            The parsed configuration.
        """
        self._everest_config = EverestConfig.model_validate(self.plan.eval(config))
        return everest2ropt(self._everest_config)

    def emit_event(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
        """
        event.data["everest_config"] = self._everest_config
        self.plan.emit_event(event)
