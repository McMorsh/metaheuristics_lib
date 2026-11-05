"""Проверки сходимости на эталонных функциях.

Пороги подобраны как защита от регрессий, а не как доказательство
сходимости: для EPO и ABC они оставлены с запасом.
"""

import itertools
from collections.abc import Callable

import pytest

from algorithms.bee_colony.artificial_bee_colony import (
    ArtificialBeeColony,
)
from algorithms.crow_search.crow_search_algorithm import (
    CrowSearchAlgorithm,
)
from algorithms.emperor_penguin.emperor_penguin_optimizer import (
    EmperorPenguinOptimizer,
)
from algorithms.grey_wolf.grey_wolf_optimizer import GreyWolfOptimizer
from algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from core.algorithm import BaseAlgorithm
from core.runner import Runner
from core.types import Objective
from problems.continuous import rastrigin, sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]

SPHERE_CASES = [
    pytest.param(ArtificialBeeColony, 1e-6, id="abc"),
    pytest.param(CrowSearchAlgorithm, 1e-6, id="csa"),
    pytest.param(EmperorPenguinOptimizer, 10.0, id="epo"),
    pytest.param(GreyWolfOptimizer, 1e-6, id="gwo"),
    pytest.param(WhaleOptimizationAlgorithm, 1e-6, id="woa"),
]

RASTRIGIN_CASES = [
    pytest.param(ArtificialBeeColony, 1e-1, id="abc"),
    pytest.param(CrowSearchAlgorithm, 1e-6, id="csa"),
    pytest.param(EmperorPenguinOptimizer, 20.0, id="epo"),
    pytest.param(GreyWolfOptimizer, 1e-6, id="gwo"),
    pytest.param(WhaleOptimizationAlgorithm, 1e-6, id="woa"),
]


def _run(
    factory: AlgorithmFactory,
    problem: Objective,
    bounds: tuple[float, float],
) -> tuple[float, list[float]]:
    algorithm = factory(
        problem,
        2,
        bounds,
        agents=30,
        max_iter=200,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result, history, _, _ = runner.run()
    return result["minimum_value"], history


@pytest.mark.parametrize(("factory", "limit"), SPHERE_CASES)
def test_algorithm_converges_on_sphere(
    factory: AlgorithmFactory,
    limit: float,
) -> None:
    value, history = _run(factory, sphere, (-10.0, 10.0))

    assert value < limit
    assert all(
        later <= earlier for earlier, later in itertools.pairwise(history)
    )


@pytest.mark.parametrize(("factory", "limit"), RASTRIGIN_CASES)
def test_algorithm_converges_on_rastrigin(
    factory: AlgorithmFactory,
    limit: float,
) -> None:
    value, history = _run(factory, rastrigin, (-5.12, 5.12))

    assert value < limit
    assert value <= history[0]
