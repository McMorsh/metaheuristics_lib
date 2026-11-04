"""Общие типы и проверка границ поиска."""

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

type Objective = Callable[[npt.NDArray[np.float64]], float]
type BoundsLike = (
    tuple[float, float]
    | Sequence[tuple[float, float]]
    | npt.NDArray[np.float64]
)


class OptimizationResult(TypedDict):
    """Результат работы алгоритма.

    Attributes:
        minimum_x: координаты лучшего найденного решения.
        minimum_value: значение целевой функции в лучшей точке.
    """

    minimum_x: list[float]
    minimum_value: float


def normalize_bounds(
    bounds: BoundsLike,
    dim: int,
) -> npt.NDArray[np.float64]:
    """Приводит границы к массиву формы ``(dim, 2)``.

    Поддерживаются общий интервал ``(low, high)``, последовательность
    интервалов длиной ``dim`` и массив NumPy соответствующей формы.

    Args:
        bounds: границы поиска.
        dim: размерность задачи.

    Returns:
        Массив нижних и верхних границ формы ``(dim, 2)``.

    Raises:
        ValueError: если границы имеют неверную форму, содержат
            нечисловые значения или ``low >= high``.
    """
    if dim < 1:
        raise ValueError(
            f"Параметр dim должен быть положительным, получено {dim}"
        )

    array = np.asarray(bounds, dtype=np.float64)

    if array.ndim == 1:
        if array.shape != (2,):
            raise ValueError(
                "Общий интервал границ должен иметь форму (2,), "
                f"получено {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Границы должны быть конечными числами")
        if array[0] >= array[1]:
            raise ValueError(
                "Нижняя граница должна быть меньше верхней, "
                f"получено {array[0]} и {array[1]}"
            )
        return np.tile(array, (dim, 1))

    if array.ndim == 2 and array.shape == (dim, 2):
        if not np.isfinite(array).all():
            raise ValueError("Границы должны быть конечными числами")
        if np.any(array[:, 0] >= array[:, 1]):
            raise ValueError(
                "Для каждой размерности нижняя граница должна быть "
                "меньше верхней"
            )
        return array.copy()

    raise ValueError(
        "Ожидались границы (low, high) или массив формы "
        f"({dim}, 2), получено {array.shape}"
    )
