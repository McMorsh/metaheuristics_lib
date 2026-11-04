"""Тесты Runner и воспроизводимости."""

import numpy as np
import numpy.typing as npt
import pytest

from algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from core.algorithm import BaseAlgorithm
from core.runner import Runner
from core.types import OptimizationResult
from problems.continuous import sphere


class BrokenAlgorithm(BaseAlgorithm):
    """Алгоритм, который всегда падает в get_result."""

    def initialize(self) -> None:
        self.best_solution = np.zeros(self.dim)
        self.best_fitness = 0.0

    def iterate(self) -> None:
        """Ничего не делает."""

    def get_result(self) -> OptimizationResult:
        raise RuntimeError("boom")


class CloseTrackingEvaluator:
    """Вычислитель, запоминающий вызов close."""

    def __init__(self) -> None:
        self.closed = False

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            [sphere(position) for position in positions],
            dtype=np.float64,
        )

    def close(self) -> None:
        self.closed = True


def _run(
    seed: int | None,
    runner_seed: int | None = None,
) -> list[float]:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=seed,
    )
    with Runner(
        algorithm,
        iteration_info=False,
        seed=runner_seed,
    ) as runner:
        _, history, _, _ = runner.run()
    return history


def test_history_has_max_iter_plus_one() -> None:
    assert len(_run(42)) == 4


def test_same_algorithm_seed_produces_same_history() -> None:
    assert _run(42) == _run(42)


def test_different_algorithm_seed_produces_different_history() -> None:
    assert _run(42) != _run(43)


def test_runner_seed_zero_is_used() -> None:
    assert _run(42, 0) == _run(42, 0)
    assert _run(42, 1) != _run(42, 0)


def test_repeated_runs_with_same_seed_are_reproducible() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        first_result, first_history, first_x, _ = runner.run()
        second_result, second_history, second_x, _ = runner.run()

    assert first_history == second_history
    assert first_result == second_result
    assert first_x == second_x


def test_runner_returns_result_and_histories() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=2,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result, history, history_x, elapsed = runner.run()

    assert set(result) == {"minimum_x", "minimum_value"}
    assert len(result["minimum_x"]) == 2
    assert len(history) == 3
    assert len(history_x) == 3
    assert all(len(point) == 2 for point in history_x)
    assert elapsed > 0.0


def test_runner_propagates_get_result_error() -> None:
    algorithm = BrokenAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=2,
        seed=42,
    )
    with pytest.raises(RuntimeError, match="boom"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_closes_algorithm() -> None:
    evaluator = CloseTrackingEvaluator()
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=2,
        seed=42,
        evaluator=evaluator,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        runner.run()
    assert evaluator.closed


def test_invalid_max_iter_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        WhaleOptimizationAlgorithm(
            sphere,
            2,
            (-5.0, 5.0),
            agents=8,
            max_iter=0,
        )
