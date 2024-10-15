"""The main k2 script."""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import click
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.simulator import Simulator
from everest.suite import (
    GRADIENT_COLUMNS,
    PERTURBATIONS_COLUMNS,
    RESULT_COLUMNS,
    SIMULATION_COLUMNS,
)
from ropt.config.plan import PlanConfig
from ropt.enums import EventType
from ropt.plan import Event, OptimizerContext, Plan
from ropt.plugins import PluginManager
from ropt.results import FunctionResults
from ruamel import yaml

from ._plugins import K2PlanPlugin

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

    output_dir = Path(everest_config.optimization_output_dir)
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        sys.exit(1)

    _add_results(plan_dict, output_dir)
    plan_config = PlanConfig.model_validate(plan_dict)

    context = OptimizerContext(
        evaluator=Simulator(everest_config),
        seed=everest_config.environment.random_seed,
    )
    plugin_manager = PluginManager()
    plugin_manager.add_plugins("plan", {"k2": K2PlanPlugin()})
    plan = Plan(plan_config, context, plugin_manager=plugin_manager)
    if verbose:
        plan.add_observer(EventType.FINISHED_EVALUATION, report)
    plan.run(everest_dict)


def _add_results(plan_dict: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    if "results" not in plan_dict:
        plan_dict["results"] = []
    for filename, columns, table_type in zip(
        ("results.txt", "gradients.txt", "simulations.txt", "perturbations.txt"),
        (
            RESULT_COLUMNS,
            GRADIENT_COLUMNS,
            SIMULATION_COLUMNS,
            PERTURBATIONS_COLUMNS,
        ),
        ("functions", "gradients", "functions", "gradients"),
        strict=True,
    ):
        plan_dict["results"].append(
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
    return plan_dict


if __name__ == "__main__":
    main()
