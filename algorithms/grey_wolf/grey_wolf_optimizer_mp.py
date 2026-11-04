"""Grey Wolf Optimizer с пулом процессов."""

from typing import Any

from algorithms._common import resolve_processes
from algorithms.grey_wolf.grey_wolf_optimizer import GreyWolfOptimizer
from core.evaluator import ProcessPoolEvaluator
from core.types import BoundsLike, Objective


class GreyWolfOptimizerMP(GreyWolfOptimizer):
    """GWO с параллельным вычислением fitness.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число волков.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        processes: число процессов, ``None`` - по числу ядер.
        process: устаревший синоним ``processes``.
        **params: дополнительные параметры GWO.
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
        process: int | None = None,
        **params: Any,
    ) -> None:
        resolved = resolve_processes(processes, process)
        self.processes = resolved
        super().__init__(
            problem,
            dim,
            bounds,
            agents,
            max_iter,
            seed,
            evaluator=ProcessPoolEvaluator(
                problem,
                max_workers=resolved,
            ),
            **params,
        )
