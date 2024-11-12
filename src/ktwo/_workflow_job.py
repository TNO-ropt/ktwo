"""This module implements the k2 workflow job step."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, List

from ert import WorkflowRunner
from ert.config import ErtConfig
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from pydantic import BaseModel, ConfigDict
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ert.storage import Storage
    from everest.config import EverestConfig
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class K2WorkflowJobStep(PlanStep):
    """The k2 workflow step."""

    class K2WorkflowJobStepWith(BaseModel):
        """Parameters used by the workflow job step.

        Attributes:
            jobs: The jobs to run.
        """

        jobs: List[str]

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(
        self,
        config: PlanStepConfig,
        plan: Plan,
        everest_config: EverestConfig,
        storage: Storage,
    ) -> None:
        """Initialize a workflow job step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        with_ = config.with_
        if isinstance(with_, str):
            with_ = {"jobs": [with_.strip()]}
        elif isinstance(with_, Sequence):
            with_ = {"jobs": with_}
        self._jobs = self.K2WorkflowJobStepWith.model_validate(with_).jobs
        self._everest_config = everest_config
        self._storage = storage

    def run(self) -> None:
        """Run the workflow job step."""
        with NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".workflow", delete=False
        ) as fp:
            file_name = Path(fp.name)
            workflow_name = file_name.stem
            for job in self._jobs:
                fp.write(
                    shlex.join([str(self.plan.eval(item)) for item in shlex.split(job)])
                    + "\n"
                )

        ert_dict = _everest_to_ert_config_dict(self._everest_config)
        try:
            ert_dict["LOAD_WORKFLOW"] = [
                *ert_dict.get("LOAD_WORKFLOW", []),
                (file_name, workflow_name),
            ]
            ert_config = ErtConfig.with_plugins().from_dict(config_dict=ert_dict)
            runner = WorkflowRunner(
                ert_config.workflows[workflow_name], self._storage, None, ert_config
            )
            runner.run_blocking()
            if not all(v["completed"] for v in runner.workflowReport().values()):
                msg = "workflow job failed"
                raise RuntimeError(msg)
        finally:
            file_name.unlink(missing_ok=True)
