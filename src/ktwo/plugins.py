"""This module implements the default optimization plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Callable, Dict, Final, Type, Union

from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from ropt.config.plan import PlanStepConfig, ResultHandlerConfig
from ropt.plan import Plan
from ropt.plugins.plan.base import PlanPlugin, PlanStep, ResultHandler

_STEP_OBJECTS: Final[Dict[str, Type[PlanStep]]] = {}
_RESULT_HANDLER_OBJECTS: Final[Dict[str, Type[ResultHandler]]] = {}


class K2PlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: Union[PlanStepConfig, ResultHandlerConfig],
        plan: Plan,
    ) -> Union[ResultHandler, PlanStep]:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_step(self, config: PlanStepConfig, plan: Plan) -> PlanStep:
        _, _, step_name = config.run.lower().rpartition("/")
        step_obj = _STEP_OBJECTS.get(step_name)
        if step_obj is not None:
            return step_obj(config, plan)

        msg = f"Unknown step type: {config.run}"
        raise TypeError(msg)

    @create.register
    def _create_result_handler(
        self, config: ResultHandlerConfig, plan: Plan
    ) -> ResultHandler:
        _, _, name = config.run.lower().rpartition("/")
        obj = _RESULT_HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(config, plan)

        msg = f"Unknown results handler object type: {config.run}"
        raise TypeError(msg)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return (method.lower() in _RESULT_HANDLER_OBJECTS) or (
            method.lower() in _STEP_OBJECTS
        )

    @property
    def data(self) -> Dict[str, Callable[..., Any]]:
        """Return plan functions implemented by the plugin.

        Returns:
            A dictionary mapping function names to callables.
        """
        return {"functions": {"everest2ropt": _everest2ropt}}


def _everest2ropt(everest_config: Dict[str, Any]) -> Dict[str, Any]:
    everest_config = EverestConfig.model_validate(everest_config)
    return everest2ropt(everest_config)
