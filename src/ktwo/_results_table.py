"""The report functionality of k2."""

from copy import deepcopy
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from ropt.config.plan import ResultHandlerConfig
from ropt.config.validated_types import ItemOrSet
from ropt.enums import EventType
from ropt.plan import Event
from ropt.plugins.plan.base import ResultHandler
from ropt.report import ResultsTable
from ropt.results import convert_to_maximize

_TABLE_TYPE_MAP: Final[dict[str, Literal["functions", "gradients"]]] = {
    "results": "functions",
    "gradients": "gradients",
    "simulations": "functions",
    "perturbations": "gradients",
}

_COLUMNS: Final[dict[str, dict[str, str]]] = {
    "results": {
        "result_id": "ID",
        "batch_id": "Batch",
        "functions.weighted_objective": "Total-Objective",
        "linear_constraints.violations": "IC-violation",
        "nonlinear_constraints.violations": "OC-violation",
        "functions.objectives": "Objective",
        "functions.constraints": "Constraint",
        "evaluations.variables": "Control",
        "linear_constraints.values": "IC-diff",
        "nonlinear_constraints.values": "OC-diff",
        "functions.scaled_objectives": "Scaled-Objective",
        "functions.scaled_constraints": "Scaled-Constraint",
        "evaluations.scaled_variables": "Scaled-Control",
        "nonlinear_constraints.scaled_values": "Scaled-OC-diff",
        "nonlinear_constraints.scaled_violations": "Scaled-OC-violation",
    },
    "gradients": {
        "result_id": "ID",
        "batch_id": "Batch",
        "gradients.weighted_objective": "Total-Gradient",
        "gradients.objectives": "Grad-objective",
        "gradients.constraints": "Grad-constraint",
    },
    "simulations": {
        "result_id": "ID",
        "batch_id": "Batch",
        "realization": "Realization",
        "evaluations.evaluation_ids": "Simulation",
        "evaluations.variables": "Control",
        "evaluations.objectives": "Objective",
        "evaluations.constraints": "Constraint",
        "evaluations.scaled_variables": "Scaled-Control",
        "evaluations.scaled_objectives": "Scaled-Objective",
        "evaluations.scaled_constraints": "Scaled-Constraint",
    },
    "perturbations": {
        "result_id": "ID",
        "batch_id": "Batch",
        "realization": "Realization",
        "evaluations.perturbed_evaluation_ids": "Simulation",
        "evaluations.perturbed_variables": "Control",
        "evaluations.perturbed_objectives": "Objective",
        "evaluations.perturbed_constraints": "Constraint",
        "evaluations.scaled_perturbed_variables": "Scaled-Control",
        "evaluations.scaled_perturbed_objectives": "Scaled-Objective",
        "evaluations.scaled_perturbed_constraints": "Scaled-Constraint",
    },
}


class K2ResultsTableHandler(ResultHandler):
    """The k2 table results handler object."""

    class K2ResultsTableHandlerWith(BaseModel):
        tags: ItemOrSet[str]
        name: str | None = None
        type_: Literal[
            "results", "gradients", "perturbations", "simulations", "defaults"
        ] = Field(default="defaults", alias="type")
        metadata: dict[str, str] = {}

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            frozen=True,
        )

    def __init__(self, config: ResultHandlerConfig, path: Path) -> None:
        self._with = self.K2ResultsTableHandlerWith.model_validate(config.with_)
        if self._with.type_ == "defaults":
            types = ["results", "gradients", "perturbations", "simulations"]
            names = [f"{type_}.txt" for type_ in types]
        else:
            types = [self._with.type_]
            names = [
                f"{self._with.type_}.txt"
                if self._with.name is None
                else self._with.name
            ]

        self._tables = []
        for type_, name in zip(types, names, strict=False):
            columns = deepcopy(_COLUMNS[type_])
            for key, title in self._with.metadata.items():
                columns[f"metadata.{key}"] = title
            self._tables.append(
                ResultsTable(
                    columns,
                    path / name,
                    table_type=_TABLE_TYPE_MAP[type_],
                    min_header_len=3,
                )
            )

    def handle_event(self, event: Event) -> Event:
        """Handle an event."""
        if (
            event.event_type
            in {
                EventType.FINISHED_EVALUATION,
                EventType.FINISHED_EVALUATOR_STEP,
            }
            and event.results is not None
            and (event.tags & self._with.tags)
        ):
            for table in self._tables:
                table.add_results(
                    event.config, (convert_to_maximize(item) for item in event.results)
                )
        return event
