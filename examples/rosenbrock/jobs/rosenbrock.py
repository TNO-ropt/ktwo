#!/usr/bin/env python

"""The rosenbrock forward model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

NVAR = 4
NREAL = 5


def rosenbrock(variables: NDArray[np.float64], realization: int) -> float:
    """Evaluate the multi-dimensional rosenbrock function."""
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=0.01, size=NREAL)
    b = rng.normal(loc=100.0, scale=1, size=NREAL)
    objective = 0.0
    for d_idx in range(NVAR - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    return -objective


def _read_point(filename: Path) -> NDArray[np.float64]:
    with filename.open("r", encoding="utf-8") as f:
        variables = json.load(f)
    return np.fromiter(
        (variables["x"][str(idx + 1)] for idx in range(NVAR)), dtype=np.float64
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vars", type=Path)
    arg_parser.add_argument("--realization", type=int)
    arg_parser.add_argument("--out", type=Path)
    options, _ = arg_parser.parse_known_args(args=sys.argv[1:])

    value = rosenbrock(_read_point(options.vars), options.realization)

    with options.out.open("w", encoding="utf-8") as f:
        f.write(f"{value:.15e}")
