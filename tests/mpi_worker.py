"""Сценарии MPI-тестов; запускается через mpiexec."""

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from metaheuristics_lib.algorithms import (
    ArtificialBeeColony,
    ArtificialBeeColonyMPI,
    CrowSearchAlgorithm,
    CrowSearchAlgorithmMPI,
    EmperorPenguinOptimizer,
    EmperorPenguinOptimizerMPI,
    GreyWolfOptimizer,
    GreyWolfOptimizerMPI,
    WhaleOptimizationAlgorithm,
    WhaleOptimizationAlgorithmMPI,
)
from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.evaluator import MPIEvaluator
from metaheuristics_lib.core.runner import Runner
from metaheuristics_lib.core.types import OptimizationResult
from metaheuristics_lib.problems.continuous import sphere

AlgorithmFactory = Callable[..., BaseAlgorithm]

COMM = MPI.COMM_WORLD
RANK = int(COMM.Get_rank())

SERIAL_MPI: dict[str, tuple[AlgorithmFactory, AlgorithmFactory]] = {
    "abc": (ArtificialBeeColony, ArtificialBeeColonyMPI),
    "csa": (CrowSearchAlgorithm, CrowSearchAlgorithmMPI),
    "epo": (EmperorPenguinOptimizer, EmperorPenguinOptimizerMPI),
    "gwo": (GreyWolfOptimizer, GreyWolfOptimizerMPI),
    "woa": (WhaleOptimizationAlgorithm, WhaleOptimizationAlgorithmMPI),
}

SCENARIOS = (*SERIAL_MPI, "error", "empty")


def _run(
    algorithm: BaseAlgorithm,
) -> tuple[OptimizationResult, list[float], list[list[float]], float]:
    with Runner(algorithm, iteration_info=False) as runner:
        return runner.run()


def check_algorithm(name: str) -> None:
    """Сравнивает serial и MPI-вариант на неравных чанках."""
    serial_factory, mpi_factory = SERIAL_MPI[name]
    kwargs: dict[str, Any] = {
        "problem": sphere,
        "dim": 2,
        "bounds": [(-1.0, 1.0), (0.0, 2.0)],
        "agents": 5,
        "max_iter": 2,
        "seed": 42,
    }
    serial = serial_factory(**kwargs)
    distributed = mpi_factory(**kwargs)

    serial_result, serial_history, _, _ = _run(serial)
    mpi_result, mpi_history, _, _ = _run(distributed)

    np.testing.assert_allclose(mpi_history, serial_history)
    np.testing.assert_allclose(
        mpi_result["minimum_x"],
        serial_result["minimum_x"],
    )
    if not np.isclose(
        mpi_result["minimum_value"],
        serial_result["minimum_value"],
    ):
        raise AssertionError("MPI и serial вернули разные значения fitness")

    if RANK == 0:
        print(f"OK {name}")


def check_error() -> None:
    """Проверяет, что ошибка objective на одном ранге не вешает MPI."""
    rank = int(COMM.Get_rank())

    def problem(position: npt.NDArray[np.float64]) -> float:
        if rank == 1:
            raise ValueError("boom on rank 1")
        return float(np.sum(position**2))

    algorithm = WhaleOptimizationAlgorithmMPI(
        problem,
        2,
        (-1.0, 1.0),
        agents=4,
        max_iter=1,
        seed=42,
    )

    try:
        algorithm.initialize()
    except RuntimeError as error:
        assert "Ошибка вычисления fitness" in str(error)
    else:
        raise AssertionError("ожидалась RuntimeError")

    if RANK == 0:
        print("OK error")


def check_empty() -> None:
    """Проверяет пустую популяцию и пустой чанк на одном из рангов."""
    with MPIEvaluator(sphere) as evaluator:
        empty = evaluator.evaluate(np.empty((0, 2)))
        single = evaluator.evaluate(np.array([[1.0, 2.0]]))

    assert empty.size == 0
    np.testing.assert_allclose(single, [5.0])

    if RANK == 0:
        print("OK empty")


def main() -> None:
    """Запускает сценарий из аргумента командной строки."""
    if len(sys.argv) != 2:
        raise SystemExit("Укажите сценарий: " + ", ".join(SCENARIOS))

    scenario = sys.argv[1]
    if scenario in SERIAL_MPI:
        check_algorithm(scenario)
    elif scenario == "error":
        check_error()
    elif scenario == "empty":
        check_empty()
    else:
        raise SystemExit(f"Неизвестный сценарий: {scenario}")


if __name__ == "__main__":
    main()
