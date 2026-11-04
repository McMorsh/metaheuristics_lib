"""Параллельное вычисление целевой функции."""

import numpy as np
import numpy.typing as npt

from core.evaluator import ProcessPoolEvaluator
from core.types import Objective


def mp_evaluate_fitness(
    positions: npt.NDArray[np.float64],
    func: Objective,
    processes: int | None = None,
) -> npt.NDArray[np.float64]:
    """Вычисляет функцию в пуле процессов.

    Функция создаёт пул на один вызов. Если вычисление выполняется
    многократно, используйте :class:`core.evaluator.ProcessPoolEvaluator`
    напрямую, чтобы переиспользовать процессы.

    Args:
        positions: массив позиций формы ``(n, dim)``.
        func: целевая функция, определённая на верхнем уровне модуля.
        processes: число процессов; ``None`` означает число ядер.

    Returns:
        Массив значений функции формы ``(n,)``.
    """
    with ProcessPoolEvaluator(
        func,
        max_workers=processes,
    ) as evaluator:
        return evaluator.evaluate(positions)
