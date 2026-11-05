"""Тесты ранней остановки и лимита вычислений."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithm,
    WhaleOptimizationAlgorithmMP,
)
from metaheuristics_lib.core.algorithm import (
    STOP_REASON_MAX_FEV,
    STOP_REASON_TARGET,
    BaseAlgorithm,
)
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]


def constant_problem(position: npt.NDArray[np.float64]) -> float:
    return 0.0


class DecreasingProblem:
    """Возвращает убывающие значения и достигает нуля на 9-м вызове."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, position: npt.NDArray[np.float64]) -> float:
        self.calls += 1
        return float(max(0, 10 - self.calls))


def _run(
    factory: AlgorithmFactory,
    problem: Callable[[npt.NDArray[np.float64]], float],
    **kwargs: Any,
) -> tuple[BaseAlgorithm, list[float]]:
    algorithm = factory(
        problem,
        2,
        (-1.0, 1.0),
        agents=4,
        max_iter=10,
        seed=42,
        **kwargs,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        _, history, _, _ = runner.run()
    return algorithm, history


def test_target_stops_during_initialization() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithm,
        constant_problem,
        target=0.0,
    )

    assert algorithm.stop_reason == STOP_REASON_TARGET
    assert algorithm.iteration == 0
    assert algorithm.nfev == algorithm.agents
    assert len(history) == 1


def test_target_stops_after_iteration() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithm,
        DecreasingProblem(),
        target=0.0,
    )

    assert algorithm.stop_reason == STOP_REASON_TARGET
    assert algorithm.iteration == 2
    assert algorithm.nfev == 12
    assert len(history) == 3


def test_max_fev_stops_before_first_iteration() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithm,
        sphere,
        max_fev=3,
    )

    assert algorithm.stop_reason == STOP_REASON_MAX_FEV
    assert algorithm.iteration == 0
    assert algorithm.nfev == algorithm.agents
    assert len(history) == 1


def test_max_fev_stops_on_iteration_boundary() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithm,
        sphere,
        max_fev=9,
    )

    assert algorithm.stop_reason == STOP_REASON_MAX_FEV
    assert algorithm.iteration == 2
    assert algorithm.nfev == 12
    assert len(history) == 3


def test_target_takes_precedence_over_max_fev() -> None:
    algorithm, _ = _run(
        WhaleOptimizationAlgorithm,
        constant_problem,
        target=0.0,
        max_fev=4,
    )

    assert algorithm.stop_reason == STOP_REASON_TARGET


def test_no_early_stop_by_default() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithm,
        sphere,
    )

    assert algorithm.stop_reason is None
    assert algorithm.iteration == algorithm.max_iter
    assert len(history) == algorithm.max_iter + 1


def test_mp_wrapper_passes_stop_parameters() -> None:
    algorithm, history = _run(
        WhaleOptimizationAlgorithmMP,
        constant_problem,
        target=0.0,
        processes=1,
    )

    assert algorithm.stop_reason == STOP_REASON_TARGET
    assert len(history) == 1


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"max_fev": 0}, ValueError, "max_fev"),
        ({"max_fev": -1}, ValueError, "max_fev"),
        ({"max_fev": 2.5}, TypeError, "max_fev"),
        ({"max_fev": True}, TypeError, "max_fev"),
        ({"target": math.nan}, ValueError, "target"),
        ({"target": math.inf}, ValueError, "target"),
        ({"target": "0"}, TypeError, "target"),
        ({"target": True}, TypeError, "target"),
    ],
)
def test_invalid_stop_parameters(
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        WhaleOptimizationAlgorithm(
            sphere,
            2,
            (-1.0, 1.0),
            agents=4,
            max_iter=10,
            seed=42,
            **kwargs,
        )
