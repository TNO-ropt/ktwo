"""The k2 run model."""

from __future__ import annotations

import pickle
import shutil
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig, ServerConfig
from everest.detached.jobs.everserver import _configure_loggers
from everest.simulator.everest_to_ert import everest_to_ert_config
from ropt.enums import EventType
from ropt.plan import Event, OptimizerContext, Plan
from ropt.plugins import PluginManager

from ._plugins import K2PlanPlugin

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config.plan import PlanConfig
    from ropt.evaluator import EvaluatorContext, EvaluatorResult


class K2RunModel(EverestRunModel):
    """The K2 run model."""

    def __init__(self, config: dict[str, Any], *, restart: bool) -> None:
        """Initialize the run model.

        Args:
            config: Everest configuration.
        """
        self._everest_config_dict = config
        everest_config = EverestConfig.model_validate(config)

        self._restart = restart
        if not self._restart:
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

        self._ert_config = everest_to_ert_config(everest_config)

        super().__init__(
            config=self._ert_config,
            everest_config=everest_config,
            simulation_callback=lambda _: None,
            optimization_callback=lambda: None,
        )

        self._restart_data: dict[str, Any] = {}

    def run_plan(
        self, plan: PlanConfig, *, report: Callable[[Event], None] | None = None
    ) -> None:
        """Run an optimization plan.

        Args:
            plan: The plan to run
        """
        self._eval_server_cfg = EvaluatorServerConfig(
            custom_port_range=range(49152, 51819)
            if self._ert_config.queue_config.queue_system == QueueSystem.LOCAL
            else None
        )
        self._experiment = next(
            (
                item
                for item in self._storage.experiments
                if item.name == self._everest_config.config_file
            ),
            None,
        )
        if self._experiment is None:
            self._experiment = self._storage.create_experiment(
                name=self._everest_config.config_file,
                parameters=self._ert_config.ensemble_config.parameter_configuration,
                responses=self._ert_config.ensemble_config.response_configuration,
            )
        plugin_manager = PluginManager()
        plugin_manager.add_plugin(
            "plan",
            "k2",
            K2PlanPlugin(self._everest_config, self._storage),
            prioritize=True,
        )
        context = OptimizerContext(
            evaluator=self._run_forward_model,
            plugin_manager=plugin_manager,
            variables={
                "config_path": str(self._everest_config.config_path.parent.resolve())
            },
        )
        context.add_observer(EventType.FINISHED_EVALUATION, self._store_restart_data)
        if report:
            context.add_observer(EventType.FINISHED_EVALUATION, report)
        Plan(plan, context).run(self._everest_config_dict)

    def _try_restart(
        self, control_values: NDArray[np.float64]
    ) -> EvaluatorResult | None:
        path = Path(self._everest_config.output_dir) / "restart"
        with (
            suppress(FileNotFoundError),
            (path / f"batch{self._batch_id}.pickle").open("rb") as file_obj,
        ):
            stored_result = pickle.load(file_obj)  # noqa: S301
            if self._batch_id == stored_result["batch_id"] and np.allclose(
                control_values, stored_result["control_values"]
            ):
                self._batch_id += 1
                return stored_result["evaluator_result"]
        return None

    def _run_forward_model(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        # Reset the current run status:
        self._restart_data = {}
        self._status = None

        # Get cached_results:
        cached_results = self._get_cached_results(control_values, evaluator_context)

        # Create the batch to run:
        batch_data = self._init_batch_data(
            control_values, evaluator_context, cached_results
        )

        # Get the evaluator result:
        evaluator_result = self._try_restart(control_values)
        if evaluator_result is None:
            assert self._experiment is not None
            ensemble_name = f"batch_{self._batch_id}"
            try:
                # Try to find an existing ensemble in storage:
                ensemble = self._experiment.get_ensemble_by_name(ensemble_name)
            except KeyError:
                # Initialize a new ensemble in storage:
                ensemble = self._experiment.create_ensemble(
                    name=ensemble_name, ensemble_size=len(batch_data)
                )
                for sim_id, controls in enumerate(batch_data.values()):
                    self._setup_sim(sim_id, controls, ensemble)

            # Get the run args:
            run_args = self._get_run_args(ensemble, evaluator_context, batch_data)

            # Remove any existing runpath directories:
            for run_arg in run_args:
                shutil.rmtree(run_arg.runpath, ignore_errors=True)

            # Evaluate the batch:
            self._context_env.update(
                {
                    "_ERT_EXPERIMENT_ID": str(ensemble.experiment_id),
                    "_ERT_ENSEMBLE_ID": str(ensemble.id),
                    "_ERT_SIMULATION_MODE": "batch_simulation",
                }
            )
            assert self._eval_server_cfg
            self._evaluate_and_postprocess(run_args, ensemble, self._eval_server_cfg)

            # If necessary, delete the run path:
            self._delete_runpath(run_args)

            # Gather the results and create the result for ropt:
            results = self._gather_simulation_results(ensemble)
            evaluator_result = self._make_evaluator_result(
                control_values, batch_data, results, cached_results
            )

            # Save restart data:
            self._restart_data = {
                "batch_id": self._batch_id,
                "control_values": control_values,
                "evaluator_result": evaluator_result,
            }

            # Increase the batch ID for the next evaluation:
            self._batch_id += 1

        # Add the results from the evaluations to the cache:
        self._add_results_to_cache(
            control_values,
            evaluator_context,
            batch_data,
            evaluator_result.objectives,
            evaluator_result.constraints,
        )

        return evaluator_result

    def _store_restart_data(self, event: Event) -> None:
        if "exit_code" not in event.data and self._restart_data:
            path = Path(self._everest_config.output_dir) / "restart"
            batch = self._restart_data["batch_id"]
            path.mkdir(exist_ok=True)
            with (path / f"batch{batch}.pickle").open("wb") as file_obj:
                pickle.dump(self._restart_data, file_obj)
