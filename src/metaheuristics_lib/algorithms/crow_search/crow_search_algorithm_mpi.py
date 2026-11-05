"""Crow Search Algorithm с MPI."""

from typing import Any

from metaheuristics_lib.algorithms.crow_search.crow_search_algorithm import (
    CrowSearchAlgorithm,
)
from metaheuristics_lib.core.evaluator import MPIEvaluator
from metaheuristics_lib.core.types import BoundsLike, Objective


class CrowSearchAlgorithmMPI(CrowSearchAlgorithm):
    """CSA с распределённым вычислением fitness.

    Все ранги выполняют одинаковые обновления состояния, а Evaluator
    распределяет вычисление функции и рассылает результаты обратно.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число ворон.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        **params: дополнительные параметры CSA.
    """

    def __init__(
        self,
        problem: Objective,
        dim: int,
        bounds: BoundsLike,
        agents: int = 100,
        max_iter: int = 50,
        seed: int | None = None,
        **params: Any,
    ) -> None:
        evaluator = MPIEvaluator(problem)
        super().__init__(
            problem,
            dim,
            bounds,
            agents,
            max_iter,
            seed=evaluator.common_seed(seed),
            evaluator=evaluator,
            **params,
        )
