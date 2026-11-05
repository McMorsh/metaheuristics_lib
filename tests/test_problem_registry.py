"""Тесты реестра тестовых задач."""

import math

import numpy as np
import pytest

import metaheuristics_lib
from metaheuristics_lib.core.types import normalize_bounds
from metaheuristics_lib.problems.registry import (
    PROBLEMS,
    ProblemSpec,
    get_problem,
    list_problems,
)

EXPECTED_NAMES = {
    "ackley",
    "griewank",
    "himmelblau",
    "mccormick",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "three_hump_camel",
}


def test_registry_contains_expected_problems() -> None:
    assert set(PROBLEMS) == EXPECTED_NAMES
    assert list_problems() == sorted(EXPECTED_NAMES)


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_registry_metadata_is_consistent(name: str) -> None:
    problem = get_problem(name)

    assert isinstance(problem, ProblemSpec)
    assert problem.name == name
    assert callable(problem.function)
    assert math.isfinite(problem.optimum)

    dim = problem.dim if problem.dim is not None else 2
    bounds = normalize_bounds(problem.bounds, dim)
    assert bounds.shape == (dim, 2)

    if problem.dim is not None:
        assert problem.optimum_x is not None
        assert len(problem.optimum_x) == problem.dim
        point = np.asarray(problem.optimum_x, dtype=np.float64)
        assert problem.function(point) == pytest.approx(
            problem.optimum,
            abs=1e-12,
        )


def test_get_problem_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Неизвестная задача"):
        get_problem("unknown")


@pytest.mark.parametrize(
    "name",
    ["ackley", "griewank", "rastrigin", "sphere"],
)
def test_variable_problems_have_zero_at_origin(name: str) -> None:
    problem = get_problem(name)
    point = np.zeros(2, dtype=np.float64)

    assert problem.dim is None
    assert problem.function(point) == pytest.approx(0.0, abs=1e-12)


def test_rosenbrock_has_zero_at_ones() -> None:
    problem = get_problem("rosenbrock")
    point = np.ones(2, dtype=np.float64)

    assert problem.dim is None
    assert problem.function(point) == pytest.approx(0.0, abs=1e-12)


def test_facade_exports_registry() -> None:
    problem = metaheuristics_lib.get_problem("sphere")

    assert problem.function is metaheuristics_lib.sphere
    assert "get_problem" in metaheuristics_lib.__all__
