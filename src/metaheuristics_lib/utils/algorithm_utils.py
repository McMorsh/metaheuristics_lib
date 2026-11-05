"""Вспомогательные функции для популяций и границ поиска."""

import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.types import (
    BoundsLike,
    Objective,
    normalize_bounds,
)

Positions = npt.NDArray[np.float64]
BoundsArray = npt.NDArray[np.float64]


def initialize_bounds(
    dim: int,
    bounds: BoundsLike,
) -> BoundsArray:
    """Приводит границы к массиву формы ``(dim, 2)``.

    Args:
        dim: размерность задачи.
        bounds: общий интервал или интервалы по размерностям.

    Returns:
        Массив границ формы ``(dim, 2)``.
    """
    return normalize_bounds(bounds, dim)


def initialize_positions(
    agents: int,
    dim: int,
    bounds: BoundsLike,
    seed: np.random.Generator | int | None = None,
) -> Positions:
    """Создаёт начальные позиции агентов.

    Args:
        agents: число агентов.
        dim: размерность задачи.
        bounds: границы поиска.
        seed: генератор, целый seed или ``None``.

    Returns:
        Массив позиций формы ``(agents, dim)``.

    Raises:
        ValueError: если ``agents`` или ``dim`` неположительны.
    """
    if agents < 1:
        raise ValueError(
            f"Параметр agents должен быть положительным, получено {agents}"
        )
    if dim < 1:
        raise ValueError(
            f"Параметр dim должен быть положительным, получено {dim}"
        )

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    bounds_array = normalize_bounds(bounds, dim)
    return rng.uniform(
        low=bounds_array[:, 0],
        high=bounds_array[:, 1],
        size=(agents, dim),
    )


def enforce_boundaries(
    positions: Positions,
    bounds: BoundsLike,
) -> Positions:
    """Ограничивает позиции границами поиска.

    Args:
        positions: массив позиций формы ``(n, dim)`` или один вектор.
        bounds: границы поиска.

    Returns:
        Массив той же формы с координатами внутри границ.
    """
    array = np.asarray(positions, dtype=np.float64)
    dim = array.shape[-1]
    bounds_array = normalize_bounds(bounds, dim)
    return np.clip(array, bounds_array[:, 0], bounds_array[:, 1])


def enforce_boundaries_csa(
    position: Positions,
    bounds: BoundsLike,
    expand_rate: float = 1.2,
) -> tuple[Positions, BoundsArray]:
    """Ограничивает вектор и при необходимости расширяет границы.

    Расширение симметрично и пропорционально ширине интервала. Исходный
    массив границ не изменяется.

    Args:
        position: вектор координат.
        bounds: границы поиска.
        expand_rate: коэффициент расширения, не меньше 1.

    Returns:
        Кортеж из ограниченного вектора и новых границ.

    Raises:
        ValueError: если ``expand_rate < 1`` или формы не совпадают.
    """
    if expand_rate < 1.0:
        raise ValueError(
            "Параметр expand_rate должен быть не меньше 1, "
            f"получено {expand_rate}"
        )

    vector = np.asarray(position, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(
            "Позиция должна быть одномерным вектором, "
            f"получена форма {vector.shape}"
        )

    bounds_array = normalize_bounds(bounds, vector.size).copy()
    width = bounds_array[:, 1] - bounds_array[:, 0]
    margin = width * (expand_rate - 1.0) / 2.0

    below = vector < bounds_array[:, 0]
    above = vector > bounds_array[:, 1]
    bounds_array[below, 0] = np.minimum(
        bounds_array[below, 0],
        vector[below] - margin[below],
    )
    bounds_array[above, 1] = np.maximum(
        bounds_array[above, 1],
        vector[above] + margin[above],
    )

    clipped = np.clip(
        vector,
        bounds_array[:, 0],
        bounds_array[:, 1],
    )
    return clipped, bounds_array


def evaluate_fitness_serial(
    positions: Positions,
    func: Objective,
) -> npt.NDArray[np.float64]:
    """Вычисляет целевую функцию для каждой позиции.

    Args:
        positions: массив позиций формы ``(n, dim)``.
        func: целевая функция.

    Returns:
        Массив значений функции формы ``(n,)``.
    """
    return np.asarray(
        [func(position) for position in positions],
        dtype=np.float64,
    )
