"""Emperor Penguin Optimizer с пулом процессов."""

from multiprocessing.context import BaseContext
from typing import Any

from metaheuristics_lib.core.evaluator import ProcessPoolEvaluator
from metaheuristics_lib.core.types import BoundsLike, Objective

from .._common import resolve_processes
from .emperor_penguin_optimizer import EmperorPenguinOptimizer


class EmperorPenguinOptimizerMP(EmperorPenguinOptimizer):
    """EPO с параллельным вычислением fitness.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число пингвинов.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        processes: число процессов, ``None`` - по числу ядер.
        process: устаревший синоним ``processes``.
        mp_context: контекст multiprocessing; ``None`` - стандартный.
        **params: дополнительные параметры EPO.
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
        mp_context: BaseContext | None = None,
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
                mp_context=mp_context,
            ),
            **params,
        )
