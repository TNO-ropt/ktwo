"""This module implements the k2 plan plugin."""

from __future__ import annotations

from functools import singledispatchmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Type

from ropt.config.plan import PlanStepConfig, ResultHandlerConfig
from ropt.plan import Plan
from ropt.plugins.plan.base import PlanPlugin, PlanStep, ResultHandler

from ._functions import _everest2ropt, _results2dict
from ._results_table import K2ResultsTableHandler
from ._workflow_job import K2WorkflowJobStep

if TYPE_CHECKING:
    from ert.storage import Storage
    from everest.config import EverestConfig

_STEP_OBJECTS: Final[dict[str, Type[PlanStep]]] = {
    "workflow_job": K2WorkflowJobStep,
}

_RESULT_HANDLER_OBJECTS: Final[dict[str, Type[ResultHandler]]] = {
    "results_table": K2ResultsTableHandler,
}


class K2PlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    def __init__(self, everest_config: EverestConfig, storage: Storage) -> None:
        self._everest_config = everest_config
        self._storage = storage

    @singledispatchmethod
    def create(  # type: ignore[override]
        self,
        config: PlanStepConfig | ResultHandlerConfig,
        plan: Plan,
    ) -> ResultHandler | PlanStep:
        """Initialize the plan plugin.

        See the [ropt.plugins.plan.base.PlanPlugin][] abstract base class.

        # noqa
        """
        msg = "Plan config type not implemented."
        raise NotImplementedError(msg)

    @create.register
    def _create_step(self, config: PlanStepConfig, plan: Plan) -> PlanStep:
        _, _, step_name = config.run.lower().rpartition("/")
        if step_name == "workflow_job":
            return K2WorkflowJobStep(config, plan, self._everest_config, self._storage)
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
        path = Path(self._everest_config.optimization_output_dir)
        if name == "results_table":
            return K2ResultsTableHandler(config, path)
        obj = _RESULT_HANDLER_OBJECTS.get(name)
        if obj is not None:
            return obj(config, plan)
        msg = f"Unknown results handler object type: {config.run}"
        raise TypeError(msg)

    def is_supported(self, method: str) -> bool:
        return (method.lower() in _RESULT_HANDLER_OBJECTS) or (
            method.lower() in _STEP_OBJECTS
        )

    @property
    def functions(self) -> dict[str, Any]:
        return {
            "everest2ropt": _everest2ropt,
            "results2dict": _results2dict,
        }
