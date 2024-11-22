"""The main k2 script."""

import sys
import warnings
from pathlib import Path
from typing import Any

import click
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from everest.simulator import Simulator
from everest.simulator.everest_to_ert import everest_to_ert_config
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import PlanConfig
from ropt.config.validated_types import ItemOrTuple
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
    plugins: ItemOrTuple[Path] = ()

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
    "--output",
    "-o",
    help="Override the output directory.",
    type=click.Path(),
    default=None,
)
def main(
    config_file: str, plan_file: str, *, verbose: bool, output: str | None
) -> None:
    """Run k2.

    k2 requires an Everest configuration file and a ropt plan file.
    """
    everest_dict = yaml_file_to_substituted_config_dict(config_file)
    if output is not None:
        everest_dict.setdefault("environment", {})
        everest_dict["environment"]["output_folder"] = output
    everest_config = EverestConfig.model_validate(everest_dict)

    config_dict = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))
    k2config = K2Config.model_validate(config_dict)

    output_dir = Path(everest_config.output_dir)
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        sys.exit(1)

    plugin_manager = PluginManager()
    ert_config = everest_to_ert_config(everest_config)
    with open_storage(ert_config.ens_path, mode="w") as storage:
        simulator = Simulator(everest_config, ert_config, storage)
        plugin_manager.add_plugins(
            "plan", {"k2": K2PlanPlugin(everest_config, storage, k2config.plugins)}
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
