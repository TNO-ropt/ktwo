"""The main k2 script."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import click
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import PlanConfig
from ruamel import yaml

from ._run_model import K2RunModel

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
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
