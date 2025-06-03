from typing import Any

import numpy as np


def enforce_boundaries_csa(x: np.ndarray, bounds: list[tuple[float, float]], expand_rate: float = 1.2) \
        -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Проверяет выход координат x за границы и корректирует их.
    Если координата выходит за пределы, соответствующий интервал границ расширяется.

    :param x: (np.ndarray) — координаты агента.
    :param bounds: (list[tuple(float, float)]) — границы по каждой координате.
    :param expand_rate: (float) — коэффициент расширения границ.

    :return:
        - np.ndarray — координаты x, приведённые к (возможно расширенным) границам.
        - list[tuple(float, float)] — обновлённые границы по каждой координате.
    """
    updated_bounds = []

    for d in range(len(x)):
        low, high = bounds[d]

        if x[d] < low:
            low = x[d] * expand_rate
        elif x[d] > high:
            high = x[d] * expand_rate

        updated_bounds.append((low, high))

    # Отдельно собираем списки нижних и верхних границ для clip
    low_b = np.array([b[0] for b in updated_bounds])
    high_b = np.array([b[1] for b in updated_bounds])

    x_clipped = np.clip(x, low_b, high_b)

    return x_clipped, updated_bounds


def enforce_boundaries(arr, bounds):
    """
    Приводит все координаты в массиве агентов к допустимым границам.

    :param arr: (np.ndarray) формы (n_agents, dim) — массив позиций агентов.
    :param bounds: Список кортежей [(low, high)] длиной dim — границы по координатам.

    :return: (np.ndarray) — массив с координатами, ограниченными в пределах границ.
    """
    low_b = np.array([b[0] for b in bounds])
    high_b = np.array([b[1] for b in bounds])

    return np.clip(arr, low_b, high_b)


def initialize_bounds(dim, bounds):
    """
    Инициализирует список границ по каждой координате.

    :param dim: int — размерность задачи.
    :param bounds:
        - tuple (low, high) — одинаковые границы для всех координат;
        - или список кортежей [(low, high), ...] длиной dim.

    :return: список кортежей [(low, high), ...] длиной dim.
    """
    if isinstance(bounds, tuple):  # один интервал
        return [bounds] * dim
    elif isinstance(bounds, list):
        if len(bounds) != dim:
            raise ValueError("Неправильное указание границ: длина bounds не совпадает с размерностью")
        return bounds
    else:
        raise TypeError("bounds должен быть кортежем или списком кортежей")


def initialize_positions(agents: int, dim: int, bounds, seed=None):
    """
    Инициализирует стартовые позиции агентов в заданных границах.

    :param agents: (int) — количество агентов.
    :param dim: (int) — размерность задачи.
    :param bounds: (tuple) или список кортежей [(low, high)].
    :param seed: (int) или None — seed для генератора случайных чисел.

    :return: (np.ndarray) формы (agents, dim) — стартовые позиции агентов.
    """
    if seed is not None:
        np.random.seed(seed)

    bounds = initialize_bounds(dim, bounds)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])

    return np.random.uniform(low=low, high=high, size=(agents, dim))


def evaluate_fitness_serial(positions: np.ndarray, func: Any) -> np.ndarray:
    """
    Вычисляет значение целевой функции для каждого агента (позиции).

    :param positions: (np.ndarray) формы (n_agents, dim) — позиции агентов.
    :param func: (Callable) — целевая функция, принимающая вектор и возвращающая float.

    :return: (np.ndarray) формы (n_agents, ...) — значения функции для каждого агента.
    """
    return np.array([func(pos) for pos in positions])
