"""Тесты вычислителей целевой функции."""

import importlib
import importlib.util
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np
import pytest

import metaheuristics_lib.core.evaluator as evaluator_module
from metaheuristics_lib.core.evaluator import (
    MPIEvaluator,
    ProcessPoolEvaluator,
    SerialEvaluator,
)
from metaheuristics_lib.problems.continuous import sphere
from metaheuristics_lib.utils.mp_utils import mp_evaluate_fitness

POSITIONS = np.array(
    [
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
EXPECTED = np.array([0.0, 5.0, 25.0])
MANY_POSITIONS = np.vstack([POSITIONS, POSITIONS])
MANY_EXPECTED = np.tile(EXPECTED, 2)


def test_serial_evaluator() -> None:
    evaluator = SerialEvaluator(sphere)
    np.testing.assert_allclose(
        evaluator.evaluate(POSITIONS),
        EXPECTED,
    )


def test_serial_evaluator_accepts_empty_positions() -> None:
    evaluator = SerialEvaluator(sphere)
    assert evaluator.evaluate(np.empty((0, 2))).size == 0


def test_serial_evaluator_rejects_vector() -> None:
    evaluator = SerialEvaluator(sphere)
    with pytest.raises(ValueError, match="двумерным"):
        evaluator.evaluate(np.array([1.0, 2.0]))


def test_process_pool_evaluator_accepts_empty_positions() -> None:
    with ProcessPoolEvaluator(sphere, max_workers=2) as evaluator:
        values = evaluator.evaluate(np.empty((0, 2)))

    assert values.size == 0


def test_process_pool_evaluator_rejects_vector() -> None:
    with (
        ProcessPoolEvaluator(sphere, max_workers=2) as evaluator,
        pytest.raises(ValueError, match="двумерным"),
    ):
        evaluator.evaluate(np.array([1.0, 2.0]))


def test_process_pool_evaluator_with_one_worker() -> None:
    with ProcessPoolEvaluator(
        sphere,
        max_workers=1,
    ) as evaluator:
        np.testing.assert_allclose(
            evaluator.evaluate(POSITIONS),
            EXPECTED,
        )


def test_process_pool_evaluator_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        ProcessPoolEvaluator(sphere, max_workers=0)


def test_process_pool_evaluator_passes_context(
    monkeypatch: pytest.MonkeyPatch,
    process_context: BaseContext | None,
) -> None:
    captured: dict[str, object] = {}

    class FakeExecutor:
        def __init__(
            self,
            max_workers: int | None = None,
            mp_context: BaseContext | None = None,
        ) -> None:
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context

        def map(
            self,
            function: Any,
            positions: Any,
            chunksize: int = 1,
        ) -> list[float]:
            return [function(position) for position in positions]

        def shutdown(self, wait: bool = True) -> None:
            """Ничего не делает."""

    monkeypatch.setattr(
        evaluator_module,
        "ProcessPoolExecutor",
        FakeExecutor,
    )

    with ProcessPoolEvaluator(
        sphere,
        max_workers=2,
        mp_context=process_context,
    ) as evaluator:
        values = evaluator.evaluate(MANY_POSITIONS)

    np.testing.assert_allclose(values, MANY_EXPECTED)
    assert captured["max_workers"] == 2
    assert captured["mp_context"] is process_context


def test_process_pool_evaluator_reuses_executor_for_larger_batches(
    monkeypatch: pytest.MonkeyPatch,
    process_context: BaseContext | None,
) -> None:
    created: list[int | None] = []

    class FakeExecutor:
        def __init__(
            self,
            max_workers: int | None = None,
            mp_context: BaseContext | None = None,
        ) -> None:
            created.append(max_workers)

        def map(
            self,
            function: Any,
            positions: Any,
            chunksize: int = 1,
        ) -> list[float]:
            return [function(position) for position in positions]

        def shutdown(self, wait: bool = True) -> None:
            """Ничего не делает."""

    monkeypatch.setattr(
        evaluator_module,
        "ProcessPoolExecutor",
        FakeExecutor,
    )
    large = np.vstack([MANY_POSITIONS] * 10)

    with ProcessPoolEvaluator(
        sphere,
        max_workers=8,
        mp_context=process_context,
    ) as evaluator:
        first = evaluator.evaluate(MANY_POSITIONS)
        second = evaluator.evaluate(large)

    np.testing.assert_allclose(first, MANY_EXPECTED)
    np.testing.assert_allclose(second, np.tile(MANY_EXPECTED, 10))
    assert created == [8]


def test_process_pool_evaluator_runs_small_batches_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_executor(*args: object, **kwargs: object) -> None:
        raise AssertionError("пул не должен создаваться для малого пакета")

    monkeypatch.setattr(
        evaluator_module,
        "ProcessPoolExecutor",
        failing_executor,
    )

    with ProcessPoolEvaluator(sphere, max_workers=2) as evaluator:
        values = evaluator.evaluate(POSITIONS)

    np.testing.assert_allclose(values, EXPECTED)


def test_process_pool_evaluator_rejects_zero_chunksize() -> None:
    with pytest.raises(ValueError, match="chunksize"):
        ProcessPoolEvaluator(sphere, chunksize=0)


def test_mp_evaluate_fitness_with_one_process() -> None:
    np.testing.assert_allclose(
        mp_evaluate_fitness(POSITIONS, sphere, processes=1),
        EXPECTED,
    )


@pytest.mark.integration
def test_mp_evaluate_fitness_accepts_context(
    process_context: BaseContext | None,
) -> None:
    np.testing.assert_allclose(
        mp_evaluate_fitness(
            MANY_POSITIONS,
            sphere,
            processes=2,
            mp_context=process_context,
        ),
        MANY_EXPECTED,
    )


@pytest.mark.integration
def test_process_pool_evaluator_with_two_workers(
    process_context: BaseContext | None,
) -> None:
    with ProcessPoolEvaluator(
        sphere,
        max_workers=2,
        mp_context=process_context,
    ) as evaluator:
        np.testing.assert_allclose(
            evaluator.evaluate(MANY_POSITIONS),
            MANY_EXPECTED,
        )


def test_mpi_evaluator_reports_missing_dependency() -> None:
    if importlib.util.find_spec("mpi4py") is not None:
        pytest.skip("mpi4py установлен")

    with pytest.raises(RuntimeError, match="mpi4py"):
        MPIEvaluator(sphere)


def test_mpi_utils_import_is_lazy() -> None:
    module = importlib.import_module("metaheuristics_lib.utils.mpi_utils")
    assert module is not None
