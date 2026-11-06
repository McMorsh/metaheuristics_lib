"""Тесты multiprocessing-обёрток алгоритмов."""

from collections.abc import Callable
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np
import pytest

from metaheuristics_lib.algorithms import (
    ArtificialBeeColony,
    ArtificialBeeColonyMP,
    CrowSearchAlgorithm,
    CrowSearchAlgorithmMP,
    EmperorPenguinOptimizer,
    EmperorPenguinOptimizerMP,
    GreyWolfOptimizer,
    GreyWolfOptimizerMP,
    WhaleOptimizationAlgorithm,
    WhaleOptimizationAlgorithmMP,
)
from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.core.types import OptimizationResult
from metaheuristics_lib.problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]


MP_PAIRS = [
    (ArtificialBeeColony, ArtificialBeeColonyMP),
    (CrowSearchAlgorithm, CrowSearchAlgorithmMP),
    (EmperorPenguinOptimizer, EmperorPenguinOptimizerMP),
    (GreyWolfOptimizer, GreyWolfOptimizerMP),
    (WhaleOptimizationAlgorithm, WhaleOptimizationAlgorithmMP),
]

ARGS: dict[str, Any] = {
    "problem": sphere,
    "dim": 2,
    "bounds": [(-1.0, 1.0), (0.0, 2.0)],
    "agents": 6,
    "max_iter": 2,
    "seed": 42,
}


def _run(algorithm: BaseAlgorithm) -> tuple[OptimizationResult, list[float]]:
    with Runner(algorithm, iteration_info=False) as runner:
        result, history, _, _ = runner.run()
    return result, history


def _assert_same_result(
    serial_result: OptimizationResult,
    serial_history: list[float],
    parallel_result: OptimizationResult,
    parallel_history: list[float],
) -> None:
    np.testing.assert_allclose(parallel_history, serial_history)
    np.testing.assert_allclose(
        parallel_result["minimum_x"],
        serial_result["minimum_x"],
    )
    assert parallel_result["minimum_value"] == pytest.approx(
        serial_result["minimum_value"]
    )


@pytest.mark.parametrize(("serial_factory", "mp_factory"), MP_PAIRS)
def test_mp_matches_serial_with_one_process(
    serial_factory: AlgorithmFactory,
    mp_factory: AlgorithmFactory,
) -> None:
    serial = serial_factory(**ARGS)
    parallel = mp_factory(**ARGS, processes=1)

    serial_result, serial_history = _run(serial)
    parallel_result, parallel_history = _run(parallel)

    _assert_same_result(
        serial_result,
        serial_history,
        parallel_result,
        parallel_history,
    )
    assert parallel.nfev == serial.nfev


@pytest.mark.integration
@pytest.mark.parametrize(("serial_factory", "mp_factory"), MP_PAIRS)
def test_mp_matches_serial_with_two_processes(
    serial_factory: AlgorithmFactory,
    mp_factory: AlgorithmFactory,
    process_context: BaseContext | None,
) -> None:
    serial = serial_factory(**ARGS)
    parallel = mp_factory(
        **ARGS,
        processes=2,
        mp_context=process_context,
    )

    serial_result, serial_history = _run(serial)
    parallel_result, parallel_history = _run(parallel)

    _assert_same_result(
        serial_result,
        serial_history,
        parallel_result,
        parallel_history,
    )
    assert parallel.nfev == serial.nfev


@pytest.mark.integration
@pytest.mark.parametrize(("serial_factory", "mp_factory"), MP_PAIRS)
def test_mp_matches_serial_with_max_fev(
    serial_factory: AlgorithmFactory,
    mp_factory: AlgorithmFactory,
    process_context: BaseContext | None,
) -> None:
    args = {**ARGS, "max_iter": 10, "max_fev": 15}
    serial = serial_factory(**args)
    parallel = mp_factory(
        **args,
        processes=2,
        mp_context=process_context,
    )

    with Runner(serial, iteration_info=False) as runner:
        serial_result, serial_history, _, _ = runner.run()
    with Runner(parallel, iteration_info=False) as runner:
        parallel_result, parallel_history, _, _ = runner.run()

    _assert_same_result(
        serial_result,
        serial_history,
        parallel_result,
        parallel_history,
    )
    assert parallel.stop_reason == serial.stop_reason
    assert parallel.nfev == serial.nfev
