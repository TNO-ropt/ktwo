"""This module implements the k2 plan plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union

from ropt.config.plan import PlanStepConfig
from ropt.plugins.plan.base import PlanPlugin, PlanStep, ResultHandler

from ._functions import fnc_everest2ropt
from ._workflow_job import K2WorkflowJobStep

if TYPE_CHECKING:
    from ert.storage import Storage
    from everest.config import EverestConfig
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Plan


class K2PlanPlugin(PlanPlugin):
    """Default plan plugin class."""

    def __init__(self, everest_config: EverestConfig, storage: Storage) -> None:
        self._everest_config = everest_config
        self._storage = storage

    def create(  # type: ignore[override]
        self,
        config: Union[PlanStepConfig, ResultHandlerConfig],
        plan: Plan,
    ) -> Union[ResultHandler, PlanStep]:
        _, _, step_name = config.run.lower().rpartition("/")
        assert step_name == "workflow_job"
        assert isinstance(config, PlanStepConfig)
        return K2WorkflowJobStep(config, plan, self._everest_config, self._storage)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        return method.lower() == "workflow_job"

    @property
    def functions(self) -> Dict[str, Any]:
        return {"everest2ropt": fnc_everest2ropt}
