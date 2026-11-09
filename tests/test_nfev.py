"""Тесты счётчика обращений к целевой функции."""

from collections.abc import Callable
from itertools import pairwise

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
from metaheuristics_lib.problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]

POPULATION_ALGORITHMS = [
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
]


def _make(
    factory: AlgorithmFactory,
    max_iter: int,
) -> BaseAlgorithm:
    return factory(
        sphere,
        2,
        (-5.0, 5.0),
        agents=6,
        max_iter=max_iter,
        seed=42,
    )


@pytest.mark.parametrize("algorithm_factory", POPULATION_ALGORITHMS)
def test_nfev_counts_initialization_and_iterations(
    algorithm_factory: AlgorithmFactory,
) -> None:
    algorithm = _make(algorithm_factory, max_iter=3)

    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()

    assert algorithm.nfev == algorithm.agents * (algorithm.max_iter + 1)


@pytest.mark.parametrize("algorithm_factory", POPULATION_ALGORITHMS)
def test_nfev_is_reset_on_repeated_run(
    algorithm_factory: AlgorithmFactory,
) -> None:
    algorithm = _make(algorithm_factory, max_iter=2)

    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()
        first_nfev = algorithm.nfev
        runner.run()

    assert algorithm.nfev == first_nfev


def test_abc_nfev_counts_employed_onlooker_and_scout_phases() -> None:
    algorithm = ArtificialBeeColony(
        sphere,
        2,
        (-5.0, 5.0),
        agents=6,
        max_iter=3,
        seed=42,
        limit=1,
    )
    base_per_iteration = 2 * algorithm.agents

    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()

    minimum = algorithm.agents + algorithm.max_iter * base_per_iteration
    assert algorithm.nfev >= minimum
    assert algorithm.nfev > minimum


def test_nfev_is_zero_before_initialization() -> None:
    algorithm = _make(WhaleOptimizationAlgorithm, max_iter=1)
    assert algorithm.nfev == 0


@pytest.mark.parametrize(
    "algorithm_factory",
    [ArtificialBeeColony, *POPULATION_ALGORITHMS],
)
def test_nfev_history_tracks_evaluations(
    algorithm_factory: AlgorithmFactory,
) -> None:
    algorithm = _make(algorithm_factory, max_iter=3)

    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()

    assert len(algorithm.nfev_history) == len(algorithm.best_fitness_history)
    assert algorithm.nfev_history[-1] == algorithm.nfev
    assert algorithm.best_fitness_history[-1] == pytest.approx(
        algorithm.best_fitness
    )
    assert all(
        later >= earlier for earlier, later in pairwise(algorithm.nfev_history)
    )
    assert all(
        later <= earlier
        for earlier, later in pairwise(algorithm.best_fitness_history)
    )


@pytest.mark.parametrize(
    "algorithm_factory",
    [ArtificialBeeColony, *POPULATION_ALGORITHMS],
)
def test_nfev_history_is_reset_on_repeated_run(
    algorithm_factory: AlgorithmFactory,
) -> None:
    algorithm = _make(algorithm_factory, max_iter=2)

    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()
        first_length = len(algorithm.nfev_history)
        runner.run()

    assert len(algorithm.nfev_history) == first_length
