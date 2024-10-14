"""The main k2 script."""

import sys
import warnings
from pathlib import Path

import click
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
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
from ropt.results import FunctionResults
from ruamel import yaml

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
def main(config_file: Path, plan_file: Path, *, verbose: bool) -> None:
    """Run the k2 script.

    The script requires an Everest configuration file and a ropt plan file.
    """
    plan_config = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))
    everest_config = EverestConfig.load_file(config_file)

    output_dir = Path(everest_config.optimization_output_dir)
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        sys.exit(1)

    if "results" not in plan_config:
        plan_config["results"] = []
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
        plan_config["results"].append(
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

    context = OptimizerContext(
        evaluator=Simulator(everest_config),
        seed=everest_config.environment.random_seed,
    )
    plan = Plan(PlanConfig.model_validate(plan_config), context)
    if verbose:
        plan.add_observer(EventType.FINISHED_EVALUATION, report)
    plan.run(everest2ropt(everest_config))


if __name__ == "__main__":
    main()
