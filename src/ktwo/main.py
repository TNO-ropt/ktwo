"""The main k2 script."""

from __future__ import annotations

import datetime
import pickle
import sys
import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, ServerConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.detached.jobs.everserver import _configure_loggers
from everest.simulator.everest_to_ert import everest_to_ert_config
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import PlanConfig
from ropt.enums import EventType
from ropt.plan import Event, OptimizerContext, Plan
from ropt.plugins import PluginManager
from ropt.results import FunctionResults, convert_to_maximize
from ruamel import yaml

from ._plugins import K2PlanPlugin

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.evaluator import EvaluatorContext, EvaluatorResult

warnings.filterwarnings("ignore")


class K2RunModel(EverestRunModel):
    """The K2 run model."""

    def __init__(self, config: dict[str, Any], *, restart: bool) -> None:
        """Initialize the run model.

        Args:
            config: Everest configuration.
        """
        self._everest_config_dict = config
        everest_config = EverestConfig.model_validate(config)

        if not restart:
            output_dir = Path(everest_config.output_dir)
            if output_dir.exists():
                print(f"Output directory exists: {output_dir}")
                sys.exit(1)

        _configure_loggers(
            detached_dir=Path(
                ServerConfig.get_detached_node_dir(everest_config.output_dir)
            ),
            log_dir=(
                Path(everest_config.output_dir) / "logs"
                if everest_config.log_dir is None
                else Path(everest_config.log_dir)
            ),
            logging_level=everest_config.logging_level,
        )

        super().__init__(
            config=everest_to_ert_config(everest_config),
            everest_config=everest_config,
            simulation_callback=lambda _: None,
            optimization_callback=lambda: None,
        )

        self._restart_data: dict[str, Any] = {}

    def run_plan(self, plan: PlanConfig, *, verbose: bool = False) -> None:
        """Run an optimization plan.

        Args:
            plan: The plan to run
        """
        self.eval_server_cfg = EvaluatorServerConfig(
            custom_port_range=range(49152, 51819)
            if self.ert_config.queue_config.queue_system == QueueSystem.LOCAL
            else None
        )
        self._experiment = self._storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",  # noqa: DTZ005
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            responses=self.ert_config.ensemble_config.response_configuration,
        )
        plugin_manager = PluginManager()
        plugin_manager.add_plugins(
            "plan", {"k2": K2PlanPlugin(self.everest_config, self._storage)}
        )
        context = OptimizerContext(
            evaluator=self._run_forward_model,
            plugin_manager=plugin_manager,
            variables={
                "config_path": str(self.everest_config.config_path.parent.resolve())
            },
        )
        context.add_observer(EventType.FINISHED_EVALUATION, self._store_restart_data)
        if verbose:
            context.add_observer(EventType.FINISHED_EVALUATION, _report)
        Plan(plan, context).run(self._everest_config_dict)

    def _run_forward_model(
        self, control_values: NDArray[np.float64], metadata: EvaluatorContext
    ) -> EvaluatorResult:
        self._restart_data = {}

        path = Path(self.everest_config.output_dir) / "restart"
        with (
            suppress(FileNotFoundError),
            (path / f"batch{self.batch_id}.pickle").open("rb") as file_obj,
        ):
            stored_result = pickle.load(file_obj)  # noqa: S301
            if self.batch_id == stored_result["batch_id"] and np.allclose(
                control_values, stored_result["control_values"]
            ):
                self.batch_id += 1
                return stored_result["evaluator_result"]

        self._restart_data = {
            "batch_id": self.batch_id,
            "control_values": control_values,
            "evaluator_result": self._forward_model_evaluator(control_values, metadata),
        }
        return self._restart_data["evaluator_result"]

    def _store_restart_data(self, event: Event) -> None:
        if event.exit_code is None and self._restart_data:
            path = Path(self.everest_config.output_dir) / "restart"
            batch = self._restart_data["batch_id"]
            path.mkdir(exist_ok=True)
            with (path / f"batch{batch}.pickle").open("wb") as file_obj:
                pickle.dump(self._restart_data, file_obj)


class K2Config(BaseModel):
    """Configuration used by the K2 program.

    Attributes:
        plan:    The plan to execute.
        plugins: Paths to plugins to load.
    """

    plan: dict[str, Any]

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        arbitrary_types_allowed=True,
        frozen=True,
    )


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("plan_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Print optimization results.")
@click.option("--restart", "-r", is_flag=True, help="Allow restart.")
def main(config_file: str, plan_file: str, *, verbose: bool, restart: bool) -> None:
    """Run k2.

    K2 requires an Everest configuration file and a K2 config file.
    """
    everest_dict = yaml_file_to_substituted_config_dict(config_file)
    k2_dict = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))
    K2RunModel(everest_dict, restart=restart).run_plan(
        PlanConfig.model_validate(K2Config.model_validate(k2_dict).plan),
        verbose=verbose,
    )


def _report(event: Event) -> None:
    """Report results of an evaluation."""
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            maximization_result = convert_to_maximize(item)
            assert maximization_result is not None
            assert isinstance(maximization_result, FunctionResults)
            print(f"result: {maximization_result.result_id}")
            print(f"  variables: {maximization_result.evaluations.variables}")
            assert maximization_result.functions is not None
            print(f"  objective: {maximization_result.functions.weighted_objective}\n")


if __name__ == "__main__":
    main()
