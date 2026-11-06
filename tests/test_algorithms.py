"""Smoke-тесты алгоритмов."""

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.algorithms import (
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
    WhaleOptimizationAlgorithmMP,
)
from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.core.types import Objective
from metaheuristics_lib.problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]

BOUNDS = [(-1.0, 1.0), (0.0, 2.0)]
LOWER = np.array([-1.0, 0.0])
UPPER = np.array([1.0, 2.0])

ALGORITHMS = [
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
]


def constant_problem(
    position: npt.NDArray[np.float64],
) -> float:
    return 1.0


@pytest.mark.parametrize("algorithm_factory", ALGORITHMS)
def test_algorithm_smoke(
    algorithm_factory: AlgorithmFactory,
) -> None:
    algorithm = algorithm_factory(
        sphere,
        2,
        BOUNDS,
        agents=8,
        max_iter=3,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result, history, history_x, _ = runner.run()

    assert len(history) == 4
    assert len(history_x) == 4
    assert len(result["minimum_x"]) == 2
    assert result["minimum_value"] == pytest.approx(
        sphere(result["minimum_x"])
    )
    assert algorithm.iteration == 3

    point = np.asarray(result["minimum_x"])
    assert np.all(point >= LOWER - 1e-12)
    assert np.all(point <= UPPER + 1e-12)

    second = algorithm_factory(
        sphere,
        2,
        BOUNDS,
        agents=8,
        max_iter=3,
        seed=42,
    )
    with Runner(second, iteration_info=False) as runner:
        second_result, second_history, _, _ = runner.run()

    assert second_history == history
    assert second_result == result


def test_mp_matches_serial_with_one_process() -> None:
    serial = WhaleOptimizationAlgorithm(
        sphere,
        2,
        BOUNDS,
        agents=8,
        max_iter=3,
        seed=42,
    )
    parallel = WhaleOptimizationAlgorithmMP(
        sphere,
        2,
        BOUNDS,
        agents=8,
        max_iter=3,
        seed=42,
        processes=1,
    )

    with Runner(serial, iteration_info=False) as runner:
        serial_result, serial_history, _, _ = runner.run()
    with Runner(parallel, iteration_info=False) as runner:
        parallel_result, parallel_history, _, _ = runner.run()

    assert parallel_history == serial_history
    assert parallel_result == serial_result


def test_gwo_requires_three_wolves() -> None:
    algorithm = GreyWolfOptimizer(
        sphere,
        2,
        BOUNDS,
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(ValueError, match="три волка"):
        algorithm.initialize()


def test_abc_requires_two_bees() -> None:
    algorithm = ArtificialBeeColony(
        sphere,
        2,
        BOUNDS,
        agents=1,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(ValueError, match="две пчелы"):
        algorithm.initialize()


def test_abc_scout_phase_uses_dimension_bounds() -> None:
    algorithm = ArtificialBeeColony(
        constant_problem,
        3,
        (-5.0, 5.0),
        agents=6,
        max_iter=4,
        seed=0,
        limit=1,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result, history, _, _ = runner.run()

    assert len(history) == 5
    assert len(result["minimum_x"]) == 3
    assert all(-5.0 <= value <= 5.0 for value in result["minimum_x"])


def test_abc_rejects_invalid_limit() -> None:
    algorithm = ArtificialBeeColony(
        sphere,
        2,
        BOUNDS,
        agents=4,
        max_iter=1,
        seed=42,
        limit=0,
    )
    with pytest.raises(ValueError, match="limit"):
        algorithm.initialize()


def test_csa_rejects_invalid_awareness() -> None:
    algorithm = CrowSearchAlgorithm(
        sphere,
        2,
        BOUNDS,
        agents=4,
        max_iter=1,
        seed=42,
        awareness_prob=2.0,
    )
    with pytest.raises(ValueError, match="awareness_prob"):
        algorithm.initialize()


def test_get_result_requires_initialization() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        BOUNDS,
        agents=8,
        max_iter=1,
        seed=42,
    )

    with pytest.raises(RuntimeError, match="не инициализирован"):
        algorithm.get_result()


def test_csa_warns_about_deprecated_expand_rate() -> None:
    algorithm = CrowSearchAlgorithm(
        sphere,
        2,
        BOUNDS,
        agents=4,
        max_iter=1,
        seed=42,
        expand_rate=1.2,
    )

    with pytest.warns(DeprecationWarning, match="expand_rate"):
        algorithm.initialize()


def test_algorithm_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        WhaleOptimizationAlgorithm(sphere, 0, BOUNDS)


def test_algorithm_rejects_float_dim() -> None:
    with pytest.raises(TypeError, match="dim"):
        WhaleOptimizationAlgorithm(
            sphere,
            cast(int, 2.5),
            BOUNDS,
        )


def test_algorithm_rejects_float_agents() -> None:
    with pytest.raises(TypeError, match="agents"):
        WhaleOptimizationAlgorithm(
            sphere,
            2,
            BOUNDS,
            agents=cast(int, 2.5),
        )


def test_algorithm_rejects_float_max_iter() -> None:
    with pytest.raises(TypeError, match="max_iter"):
        WhaleOptimizationAlgorithm(
            sphere,
            2,
            BOUNDS,
            max_iter=cast(int, 2.5),
        )


def test_algorithm_rejects_invalid_agents() -> None:
    with pytest.raises(ValueError, match="agents"):
        WhaleOptimizationAlgorithm(sphere, 2, BOUNDS, agents=0)


def test_algorithm_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="Нижняя граница"):
        WhaleOptimizationAlgorithm(sphere, 2, (1.0, 1.0))


def test_algorithm_rejects_non_callable_problem() -> None:
    with pytest.raises(TypeError, match="problem"):
        WhaleOptimizationAlgorithm(
            cast(Objective, 1),
            2,
            BOUNDS,
        )


def test_algorithm_rejects_unknown_parameter() -> None:
    with pytest.raises(TypeError, match="Неизвестные параметры"):
        WhaleOptimizationAlgorithm(sphere, 2, BOUNDS, typo=1)


def test_mp_wrapper_rejects_unknown_parameter() -> None:
    with pytest.raises(TypeError, match="typo"):
        WhaleOptimizationAlgorithmMP(
            sphere,
            2,
            BOUNDS,
            processes=1,
            typo=1,
        )
