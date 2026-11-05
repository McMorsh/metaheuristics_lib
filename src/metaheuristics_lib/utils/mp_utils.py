"""Параллельное вычисление целевой функции."""

from multiprocessing.context import BaseContext

import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.evaluator import ProcessPoolEvaluator
from metaheuristics_lib.core.types import Objective


def mp_evaluate_fitness(
    positions: npt.NDArray[np.float64],
    func: Objective,
    processes: int | None = None,
    *,
    mp_context: BaseContext | None = None,
) -> npt.NDArray[np.float64]:
    """Вычисляет функцию в пуле процессов.

    Функция создаёт пул на один вызов. Если вычисление выполняется
    многократно, используйте :class:`core.evaluator.ProcessPoolEvaluator`
    напрямую, чтобы переиспользовать процессы.

    Args:
        positions: массив позиций формы ``(n, dim)``.
        func: целевая функция, определённая на верхнем уровне модуля.
        processes: число процессов; ``None`` означает число ядер.
        mp_context: контекст multiprocessing; ``None`` означает стандартный.

    Returns:
        Массив значений функции формы ``(n,)``.
    """
    with ProcessPoolEvaluator(
        func,
        max_workers=processes,
        mp_context=mp_context,
    ) as evaluator:
        return evaluator.evaluate(positions)
