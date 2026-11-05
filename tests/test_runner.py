"""Тесты Runner и воспроизводимости."""

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.core.types import OptimizationResult
from metaheuristics_lib.problems.continuous import sphere


class BrokenAlgorithm(BaseAlgorithm):
    """Алгоритм, который всегда падает в get_result."""

    def initialize(self) -> None:
        self.best_solution = np.zeros(self.dim)
        self.best_fitness = 0.0

    def iterate(self) -> None:
        """Ничего не делает."""

    def get_result(self) -> OptimizationResult:
        raise RuntimeError("boom")


class StubAlgorithm(BaseAlgorithm):
    """Алгоритм без реальных итераций для проверки Runner."""

    def initialize(self) -> None:
        self.best_solution = np.zeros(self.dim)
        self.best_fitness = 0.0

    def iterate(self) -> None:
        """Ничего не делает."""


class NoneResultAlgorithm(StubAlgorithm):
    """Алгоритм, возвращающий None вместо результата."""

    def get_result(self) -> OptimizationResult:
        return cast(OptimizationResult, None)


class MissingKeyAlgorithm(StubAlgorithm):
    """Алгоритм с неполным результатом."""

    def get_result(self) -> OptimizationResult:
        return cast(OptimizationResult, {"minimum_x": [0.0, 0.0]})


class WrongValueAlgorithm(StubAlgorithm):
    """Алгоритм с неверным типом значения."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": [0.0, 0.0], "minimum_value": "bad"},
        )


class StringCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм с нечисловыми координатами."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": ["1.0", "2.0"], "minimum_value": 0.0},
        )


class NonIterableCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм с неитерируемыми координатами."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": 1.0, "minimum_value": 0.0},
        )


class MappingCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм с координатами в виде отображения."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {
                "minimum_x": {1.0: "a", 2.0: "b"},
                "minimum_value": 0.0,
            },
        )


class SetCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм с координатами в неупорядоченном множестве."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": {1.0, 2.0}, "minimum_value": 0.0},
        )


class StringContainerCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм со строкой вместо последовательности координат."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": "12", "minimum_value": 0.0},
        )


class ShortCoordinatesAlgorithm(StubAlgorithm):
    """Алгоритм с координатами неверной длины."""

    def get_result(self) -> OptimizationResult:
        return cast(
            OptimizationResult,
            {"minimum_x": [0.0], "minimum_value": 0.0},
        )


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


def test_runner_rejects_none_result() -> None:
    algorithm = NoneResultAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="словарь"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_missing_result_key() -> None:
    algorithm = MissingKeyAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(KeyError, match="minimum_value"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_wrong_result_type() -> None:
    algorithm = WrongValueAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="minimum_value"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_non_numeric_coordinates() -> None:
    algorithm = StringCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="только числа"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_non_iterable_coordinates() -> None:
    algorithm = NonIterableCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="последовательностью"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_mapping_coordinates() -> None:
    algorithm = MappingCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="последовательностью"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_set_coordinates() -> None:
    algorithm = SetCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="последовательностью"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_string_container_coordinates() -> None:
    algorithm = StringContainerCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(TypeError, match="последовательностью"):
        Runner(algorithm, iteration_info=False).run()


def test_runner_rejects_wrong_coordinate_count() -> None:
    algorithm = ShortCoordinatesAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
    )
    with pytest.raises(ValueError, match="2 координат"):
        Runner(algorithm, iteration_info=False).run()


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


def test_runner_closes_algorithm_without_context() -> None:
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
    Runner(algorithm, iteration_info=False).run()
    assert evaluator.closed


def test_runner_closes_algorithm_on_error() -> None:
    evaluator = CloseTrackingEvaluator()
    algorithm = BrokenAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=2,
        max_iter=1,
        seed=42,
        evaluator=evaluator,
    )
    with pytest.raises(RuntimeError, match="boom"):
        Runner(algorithm, iteration_info=False).run()
    assert evaluator.closed


def test_runner_import_does_not_configure_logging() -> None:
    import metaheuristics_lib.core.runner as runner_module

    runner_logger = logging.getLogger(runner_module.__name__)
    assert runner_logger.handlers == []
    assert runner_logger.propagate is True


def test_invalid_max_iter_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        WhaleOptimizationAlgorithm(
            sphere,
            2,
            (-5.0, 5.0),
            agents=8,
            max_iter=0,
        )


def test_runner_rejects_algorithm_with_mutated_max_iter() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=1,
        seed=42,
    )
    algorithm.max_iter = 0

    with pytest.raises(ValueError, match="max_iter"):
        Runner(algorithm, iteration_info=False).run()
