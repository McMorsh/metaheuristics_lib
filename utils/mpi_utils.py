"""MPI-вычисление целевой функции.

Модуль не импортирует ``mpi4py`` на уровне модуля: зависимость
загружается только при создании :class:`core.evaluator.MPIEvaluator`.
"""

import numpy as np
import numpy.typing as npt

from core.evaluator import MPIEvaluator
from core.types import Objective

__all__ = ["MPIEvaluator", "mpi_evaluate_fitness"]


def mpi_evaluate_fitness(
    positions: npt.NDArray[np.float64],
    func: Objective,
) -> npt.NDArray[np.float64]:
    """Вычисляет функцию на всех MPI-рангах.

    Args:
        positions: массив позиций формы ``(n, dim)``.
        func: целевая функция.

    Returns:
        Полный массив значений функции на каждом ранге.
    """
    evaluator = MPIEvaluator(func)
    return evaluator.evaluate(positions)
