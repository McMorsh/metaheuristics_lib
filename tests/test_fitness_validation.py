"""Тесты проверки конечности fitness."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.algorithms import (
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
)
from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.runner import Runner

AlgorithmFactory = Callable[..., BaseAlgorithm]

ALGORITHMS = [
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
]

NON_FINITE_VALUES = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="negative_inf"),
]


@pytest.mark.parametrize("algorithm_factory", ALGORITHMS)
@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_algorithm_rejects_non_finite_fitness(
    algorithm_factory: AlgorithmFactory,
    value: float,
) -> None:
    def problem(position: npt.NDArray[np.float64]) -> float:
        return value

    algorithm = algorithm_factory(
        problem,
        2,
        (-1.0, 1.0),
        agents=4,
        max_iter=1,
        seed=42,
    )

    with (
        Runner(algorithm, iteration_info=False) as runner,
        pytest.raises(ValueError, match="конечными"),
    ):
        runner.run()

    assert algorithm.nfev == algorithm.agents
    assert algorithm.best_solution is None
    assert algorithm.best_fitness == math.inf


def test_non_finite_fitness_reports_position() -> None:
    calls = 0

    def problem(position: npt.NDArray[np.float64]) -> float:
        nonlocal calls
        calls += 1
        return math.nan if calls == 2 else 0.0

    algorithm = WhaleOptimizationAlgorithm(
        problem,
        2,
        (-1.0, 1.0),
        agents=4,
        max_iter=1,
        seed=42,
    )

    with pytest.raises(ValueError, match="позиции 1"):
        algorithm.initialize()
