"""The main k2 script."""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import click
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.simulator import Simulator
from everest.simulator.everest_to_ert import everest_to_ert_config
from ropt.config.plan import PlanConfig
from ropt.enums import EventType
from ropt.plan import Event, OptimizerContext, Plan
from ropt.plugins import PluginManager
from ropt.results import FunctionResults
from ruamel import yaml

from ktwo.plugins import K2PlanPlugin

warnings.filterwarnings("ignore")


def report(event: Event) -> None:
    """Report results of an evaluation."""
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"result: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("plan_file", type=click.Path(exists=True))
@click.option("--verbose", "-v/ ", is_flag=True, help="Print optimization results.")
def main(config_file: str, plan_file: str, *, verbose: bool) -> None:
    """Run the k2 script.

    The script requires an Everest configuration file and a ropt plan file.
    """
    everest_dict = yaml_file_to_substituted_config_dict(config_file)
    plan_dict = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))

    everest_config = EverestConfig.model_validate(everest_dict)

    output_dir = Path(everest_config.output_dir)
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        sys.exit(1)
    _add_results(plan_dict["plan"], output_dir)

    ert_config = everest_to_ert_config(everest_config)
    with open_storage(ert_config.ens_path, mode="w") as storage:
        context = OptimizerContext(
            evaluator=Simulator(everest_config, ert_config, storage),
            seed=everest_config.environment.random_seed,
        )
        plugin_manager = PluginManager()
        plugin_manager.add_plugins("plan", {"k2": K2PlanPlugin()})
        plan = Plan(
            PlanConfig.model_validate(plan_dict["plan"]),
            context,
            plugin_manager=plugin_manager,
        )
        if verbose:
            plan.add_observer(EventType.FINISHED_EVALUATION, report)
        plan.run(everest_dict)


def _add_results(plan: Dict[str, Any], output_dir: Path) -> None:
    result_columns = {
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
    }
    gradient_columns = {
        "result_id": "ID",
        "batch_id": "Batch",
        "gradients.weighted_objective": "Total-Gradient",
        "gradients.objectives": "Grad-objective",
        "gradients.constraints": "Grad-constraint",
    }
    simulation_columns = {
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
    }
    perturbations_columns = {
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
    }

    if "results" not in plan:
        plan["results"] = []
    for filename, columns, table_type in zip(
        ("results.txt", "gradients.txt", "simulations.txt", "perturbations.txt"),
        (
            result_columns,
            gradient_columns,
            simulation_columns,
            perturbations_columns,
        ),
        ("functions", "gradients", "functions", "gradients"),
        strict=True,
    ):
        plan["results"].append(
            {
                "run": "table",
                "with": {
                    "tags": "report",
                    "columns": columns,
                    "path": output_dir / filename,
                    "table_type": table_type,
                    "maximize": True,
                },
            }
        )


if __name__ == "__main__":
    main()
