"""The main k2 script."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import PlanConfig
from ropt.results import FunctionResults, convert_to_maximize
from ruamel import yaml

from ._run_model import K2RunModel

if TYPE_CHECKING:
    from ropt.plan import Event

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
@click.option("--restart", "-r", is_flag=True, help="Allow restart.")
def main(config_file: str, plan_file: str, *, verbose: bool, restart: bool) -> None:
    """Run k2.

    K2 requires an Everest configuration file and a K2 config file.
    """
    everest_dict = yaml_file_to_substituted_config_dict(config_file)
    k2_dict = yaml.YAML(typ="safe", pure=True).load(Path(plan_file))
    K2RunModel(everest_dict, restart=restart).run_plan(
        PlanConfig.model_validate(K2Config.model_validate(k2_dict).plan),
        report=_report if verbose else None,
    )


def _report(event: Event) -> None:
    """Report results of an evaluation."""
    for item in event.data["results"]:
        if isinstance(item, FunctionResults) and item.functions is not None:
            maximization_result = convert_to_maximize(item)
            assert maximization_result is not None
            assert isinstance(maximization_result, FunctionResults)
            print(f"  variables: {maximization_result.evaluations.variables}")
            assert maximization_result.functions is not None
            print(f"  objective: {maximization_result.functions.weighted_objective}\n")


if __name__ == "__main__":
    main()
