"""The main k2 script."""

import sys
import warnings
from pathlib import Path
from typing import Any

import click
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.simulator import Simulator
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import PlanConfig
from ropt.enums import EventType
from ropt.plan import Event, OptimizerContext, Plan
from ropt.plugins import PluginManager
from ropt.results import FunctionResults, convert_to_maximize
from ruamel import yaml

from ._plugins import K2PlanPlugin

warnings.filterwarnings("ignore")


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
@click.option(
    "--functions",
    "-f",
    type=click.Path(),
    help="Path to user functions.",
    multiple=True,
)
def main(
    config_file: str,
    plan_file: str,
    functions: tuple[str, ...],
    *,
    verbose: bool,
) -> None:
    """Run k2.

    k2 requires an Everest configuration file and a ropt plan file.
    """
    everest_dict = yaml_file_to_substituted_config_dict(config_file)
    everest_config = EverestConfig.model_validate(everest_dict)

    config_dict = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))
    k2config = K2Config.model_validate(config_dict)

    output_dir = Path(everest_config.output_dir)
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        sys.exit(1)

    run_model = EverestRunModel.create(everest_config)
    simulator = Simulator(
        run_model.everest_config,
        run_model.ert_config,
        run_model._storage,  # noqa: SLF001
    )
    plugin_manager = PluginManager()
    plugin_manager.add_plugins(
        "plan",
        {"k2": K2PlanPlugin(run_model.everest_config, run_model._storage, functions)},  # noqa: SLF001
    )
    context = OptimizerContext(
        evaluator=simulator.create_forward_model_evaluator_function(),
        plugin_manager=plugin_manager,
        variables={"config_path": str(everest_config.config_path.parent.resolve())},
    )
    if verbose:
        context.add_observer(EventType.FINISHED_EVALUATION, _report)
    plan = Plan(PlanConfig.model_validate(k2config.plan), context)
    plan.run(everest_dict)


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
