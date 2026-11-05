"""Тесты multiprocessing-обёрток алгоритмов."""

from collections.abc import Callable
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np
import pytest

from algorithms._common import resolve_processes
from algorithms.bee_colony.artificial_bee_colony import ArtificialBeeColony
from algorithms.bee_colony.artificial_bee_colony_mp import (
    ArtificialBeeColonyMP,
)
from algorithms.crow_search.crow_search_algorithm import CrowSearchAlgorithm
from algorithms.crow_search.crow_search_algorithm_mp import (
    CrowSearchAlgorithmMP,
)
from algorithms.emperor_penguin.emperor_penguin_optimizer import (
    EmperorPenguinOptimizer,
)
from algorithms.emperor_penguin.emperor_penguin_optimizer_mp import (
    EmperorPenguinOptimizerMP,
)
from algorithms.grey_wolf.grey_wolf_optimizer import GreyWolfOptimizer
from algorithms.grey_wolf.grey_wolf_optimizer_mp import GreyWolfOptimizerMP
from algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from algorithms.whale.whale_optimization_algorithm_mp import (
    WhaleOptimizationAlgorithmMP,
)
from core.algorithm import BaseAlgorithm
from core.runner import Runner
from core.types import OptimizationResult
from problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]


def test_resolve_processes_prefers_new_name() -> None:
    assert resolve_processes(2, None) == 2


def test_resolve_processes_supports_deprecated_name() -> None:
    with pytest.warns(DeprecationWarning, match="process"):
        assert resolve_processes(None, 3) == 3


def test_resolve_processes_rejects_conflict() -> None:
    with (
        pytest.warns(DeprecationWarning, match="process"),
        pytest.raises(ValueError, match="противоречат"),
    ):
        resolve_processes(2, 3)


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
