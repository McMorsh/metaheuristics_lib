"""Artificial Bee Colony с пулом процессов."""

from multiprocessing.context import BaseContext
from typing import Any

from metaheuristics_lib.algorithms.bee_colony.artificial_bee_colony import (
    ArtificialBeeColony,
)
from metaheuristics_lib.core.evaluator import ProcessPoolEvaluator
from metaheuristics_lib.core.types import BoundsLike, Objective


class ArtificialBeeColonyMP(ArtificialBeeColony):
    """ABC с параллельным вычислением fitness.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число пчёл.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        processes: число процессов, ``None`` - по числу ядер.
        mp_context: контекст multiprocessing; ``None`` - стандартный.
        **params: дополнительные параметры ABC.
    """

    def __init__(
        self,
        problem: Objective,
        dim: int,
        bounds: BoundsLike,
        agents: int = 100,
        max_iter: int = 50,
        seed: int | None = None,
        *,
        processes: int | None = None,
        mp_context: BaseContext | None = None,
        **params: Any,
    ) -> None:
        self.processes = processes
        super().__init__(
            problem,
            dim,
            bounds,
            agents,
            max_iter,
            seed,
            evaluator=ProcessPoolEvaluator(
                problem,
                max_workers=processes,
                mp_context=mp_context,
            ),
            **params,
        )
