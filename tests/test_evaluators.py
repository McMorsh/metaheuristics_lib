"""Тесты вычислителей целевой функции."""

import importlib
import importlib.util
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np
import pytest

import core.evaluator as evaluator_module
from core.evaluator import (
    MPIEvaluator,
    ProcessPoolEvaluator,
    SerialEvaluator,
)
from problems.continuous import sphere
from utils.mp_utils import mp_evaluate_fitness

POSITIONS = np.array(
    [
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
EXPECTED = np.array([0.0, 5.0, 25.0])


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
        values = evaluator.evaluate(POSITIONS)

    np.testing.assert_allclose(values, EXPECTED)
    assert captured["max_workers"] == 2
    assert captured["mp_context"] is process_context


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
            POSITIONS,
            sphere,
            processes=2,
            mp_context=process_context,
        ),
        EXPECTED,
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
            evaluator.evaluate(POSITIONS),
            EXPECTED,
        )


def test_mpi_evaluator_reports_missing_dependency() -> None:
    if importlib.util.find_spec("mpi4py") is not None:
        pytest.skip("mpi4py установлен")

    with pytest.raises(RuntimeError, match="mpi4py"):
        MPIEvaluator(sphere)


def test_mpi_utils_import_is_lazy() -> None:
    module = importlib.import_module("utils.mpi_utils")
    assert module is not None
