"""Тесты непрерывных функций."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.problems.continuous import (
    ackley,
    griewank,
    himmelblau,
    mccormick,
    rastrigin,
    rosenbrock,
    sphere,
    three_hump_camel,
)

Function = Callable[[npt.ArrayLike], float]


@pytest.mark.parametrize(
    ("function", "point", "expected"),
    [
        (sphere, [0.0, 0.0], 0.0),
        (sphere, [1.0, 2.0], 5.0),
        (rastrigin, [0.0, 0.0], 0.0),
        (ackley, [0.0, 0.0], 0.0),
        (griewank, [0.0, 0.0], 0.0),
        (rosenbrock, [1.0, 1.0], 0.0),
        (himmelblau, [3.0, 2.0], 0.0),
        (
            mccormick,
            [-0.54719, -1.54719],
            -1.913222954981037,
        ),
        (three_hump_camel, [0.0, 0.0], 0.0),
    ],
)
def test_functions_return_expected_values(
    function: Function,
    point: list[float],
    expected: float,
) -> None:
    assert function(point) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "function",
    [sphere, rastrigin, ackley, griewank],
)
def test_functions_reject_empty_vector(function: Function) -> None:
    with pytest.raises(ValueError, match="не должен быть пустым"):
        function(np.array([]))


@pytest.mark.parametrize(
    ("function", "size", "message"),
    [
        (mccormick, 1, "размерностью 2"),
        (himmelblau, 1, "размерностью 2"),
        (three_hump_camel, 1, "размерностью 2"),
        (rosenbrock, 1, "меньше 2"),
    ],
)
def test_functions_validate_size(
    function: Function,
    size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(np.zeros(size))


def test_functions_reject_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="одномерным"):
        sphere(np.zeros((2, 2)))
