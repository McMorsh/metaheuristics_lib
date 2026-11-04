"""Тесты вычислителей целевой функции."""

import importlib
import importlib.util

import numpy as np
import pytest

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


def test_mp_evaluate_fitness_with_one_process() -> None:
    np.testing.assert_allclose(
        mp_evaluate_fitness(POSITIONS, sphere, processes=1),
        EXPECTED,
    )


@pytest.mark.integration
def test_process_pool_evaluator_with_two_workers() -> None:
    with ProcessPoolEvaluator(
        sphere,
        max_workers=2,
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
