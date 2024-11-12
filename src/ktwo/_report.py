"""The report functionality of k2."""

from pathlib import Path
from typing import Any, Dict


def add_results(config: Dict[str, Any], output_dir: Path) -> None:
    plan = config["plan"]
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

    metadata = config.get("report", {}).get("metadata", {})
    for key, title in metadata.items():
        result_columns[f"metadata.{key}"] = title

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
