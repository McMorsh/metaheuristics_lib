"""Тесты метрик сходимости."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from metaheuristics_lib.algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.problems.continuous import sphere
from metaheuristics_lib.utils.metrics import (
    MultipleRunResults,
    area_under_curve,
    best_at_budget,
    best_so_far,
    evaluations_to_target,
    mean_best,
    run_repeated,
    summarize_run,
    time_to_target,
)


def _make_algorithm(
    seed: int,
) -> WhaleOptimizationAlgorithm:
    return WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=seed,
    )


def _run_once(seed: int) -> tuple[float, list[float]]:
    with Runner(
        _make_algorithm(seed),
        iteration_info=False,
    ) as runner:
        result, history, _, _ = runner.run()
    return result["minimum_value"], history


def test_best_so_far_accumulates_minimum() -> None:
    assert best_so_far([3.0, 2.0, 4.0, 1.0]) == [
        3.0,
        2.0,
        2.0,
        1.0,
    ]


def test_best_so_far_accepts_empty_history() -> None:
    assert best_so_far([]) == []


def test_best_so_far_rejects_matrix() -> None:
    with pytest.raises(ValueError, match="одномерной"):
        best_so_far(np.array([[1.0, 2.0]]))


def test_area_under_curve_normalizes() -> None:
    assert area_under_curve([1.0, 1.0, 1.0]) == pytest.approx(2.0 / 3.0)


def test_area_under_curve_accepts_numpy_history() -> None:
    assert area_under_curve(
        np.array([1.0, 1.0, 1.0]),
    ) == pytest.approx(2.0 / 3.0)


def test_area_under_curve_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="пустой"):
        area_under_curve([])


def test_area_under_curve_rejects_matrix() -> None:
    with pytest.raises(ValueError, match="одномерной"):
        area_under_curve(np.array([[1.0, 2.0]]))


def test_time_to_target_returns_first_index() -> None:
    assert time_to_target([5.0, 4.0, 3.0], 4.0) == 1


def test_time_to_target_accepts_numpy_history() -> None:
    assert (
        time_to_target(
            np.array([5.0, 4.0, 3.0]),
            4.0,
        )
        == 1
    )


def test_time_to_target_returns_none() -> None:
    assert time_to_target([5.0], 1.0) is None


def test_time_to_target_rejects_matrix() -> None:
    with pytest.raises(ValueError, match="одномерной"):
        time_to_target(np.array([[5.0, 4.0]]), 4.0)


def test_mean_best() -> None:
    assert mean_best([[1.0, 2.0], [3.0, 4.0]]) == 2.0


def test_mean_best_accepts_numpy_histories() -> None:
    assert mean_best([np.array([1.0, 2.0]), np.array([3.0, 4.0])]) == 2.0


def test_mean_best_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="пустым"):
        mean_best([])


def test_mean_best_rejects_matrix_history() -> None:
    with pytest.raises(ValueError, match="одномерной"):
        mean_best([np.array([[1.0, 2.0]])])


def test_mean_best_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="пустой"):
        mean_best([[]])


def test_run_repeated_returns_expected_structure() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result = run_repeated(runner, runs=2, seed=42)

    assert len(result["best_fitness"]) == 2
    assert len(result["history_of_best_fitness"]) == 2
    assert len(result["best_x"]) == 2
    assert len(result["history_of_best_x"]) == 2
    assert len(result["history_of_best_x"][0]) == 4
    assert len(result["time"]) == 2
    assert (
        result["history_of_best_fitness"][0]
        != result["history_of_best_fitness"][1]
    )


def test_run_repeated_uses_seed_sequence() -> None:
    with Runner(
        _make_algorithm(42),
        iteration_info=False,
    ) as runner:
        result = run_repeated(runner, runs=2)

    first_value, first_history = _run_once(42)
    second_value, second_history = _run_once(43)

    assert first_history != second_history
    assert result["history_of_best_fitness"][0] == first_history
    assert result["history_of_best_fitness"][1] == second_history
    assert result["best_fitness"] == pytest.approx([first_value, second_value])


def test_run_repeated_restores_seed_state() -> None:
    with Runner(
        _make_algorithm(42),
        iteration_info=False,
    ) as runner:
        original_runner_seed = runner.seed
        run_repeated(runner, runs=2)

        assert runner.seed == original_runner_seed
        assert runner.algorithm.seed == 42


def test_run_repeated_rejects_zero_runs() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=42,
    )
    with (
        Runner(algorithm, iteration_info=False) as runner,
        pytest.raises(ValueError, match="runs"),
    ):
        run_repeated(runner, runs=0)


def test_summarize_run_returns_statistics() -> None:
    algorithm = WhaleOptimizationAlgorithm(
        sphere,
        2,
        (-5.0, 5.0),
        agents=8,
        max_iter=3,
        seed=42,
    )
    with Runner(algorithm, iteration_info=False) as runner:
        result = run_repeated(runner, runs=2, seed=42)

    summary = summarize_run(result, target=0.0)
    assert set(summary) == {
        "mean_final",
        "std_final",
        "mean_auc",
        "std_auc",
        "time",
        "mean_time_to_target",
        "success_rate",
    }


def test_summarize_run_rejects_empty_results() -> None:
    result = MultipleRunResults(
        best_fitness=[],
        history_of_best_fitness=[],
        best_x=[],
        history_of_best_x=[],
        time=[],
    )

    with pytest.raises(ValueError, match="пустыми"):
        summarize_run(result)


def test_summarize_run_rejects_empty_history() -> None:
    result = MultipleRunResults(
        best_fitness=[1.0],
        history_of_best_fitness=[[]],
        best_x=[],
        history_of_best_x=[],
        time=[1.0],
    )

    with pytest.raises(ValueError, match="пустой"):
        summarize_run(result)


def test_summarize_run_rejects_mismatched_lengths() -> None:
    result = MultipleRunResults(
        best_fitness=[1.0],
        history_of_best_fitness=[[1.0]],
        best_x=[],
        history_of_best_x=[],
        time=[1.0, 2.0],
    )

    with pytest.raises(ValueError, match="одинаковую длину"):
        summarize_run(result)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (4, None),
        (5, 10.0),
        (12, 8.0),
        (100, 6.0),
    ],
)
def test_best_at_budget_returns_last_point(
    budget: int,
    expected: float | None,
) -> None:
    value = best_at_budget(
        [5, 10, 15],
        [10.0, 8.0, 6.0],
        budget,
    )

    assert value == expected


def test_best_at_budget_accepts_numpy_histories() -> None:
    value = best_at_budget(
        np.array([5, 10, 15]),
        np.array([10.0, 8.0, 6.0]),
        10,
    )

    assert value == pytest.approx(8.0)


def test_best_at_budget_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        best_at_budget([5], [1.0], -1)


def test_best_at_budget_rejects_non_integer_budget() -> None:
    with pytest.raises(TypeError, match="budget"):
        best_at_budget(
            [5],
            [1.0],
            1.5,  # type: ignore[arg-type]  # проверка runtime
        )


@pytest.mark.parametrize(
    ("nfev", "best"),
    [
        ([], []),
        ([5, 10], [1.0]),
        ([10, 5], [1.0, 2.0]),
        (np.array([[5]]), [1.0]),
    ],
)
def test_best_at_budget_rejects_invalid_histories(
    nfev: Sequence[int] | npt.NDArray[np.int64],
    best: Sequence[float],
) -> None:
    with pytest.raises(ValueError):
        best_at_budget(nfev, best, 5)


def test_evaluations_to_target_returns_first_hit() -> None:
    value = evaluations_to_target(
        [5, 10, 15],
        [10.0, 8.0, 6.0],
        8.0,
    )

    assert value == 10


def test_evaluations_to_target_returns_none() -> None:
    value = evaluations_to_target(
        [5, 10, 15],
        [10.0, 8.0, 6.0],
        5.0,
    )

    assert value is None


def test_evaluations_to_target_uses_tolerance() -> None:
    value = evaluations_to_target(
        [5],
        [1.0 + 1e-10],
        1.0,
    )

    assert value == 5
