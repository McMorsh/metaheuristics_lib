from random import random
from typing import Any

import numpy as np


def enforce_boundaries(x, low_b, high_b, expand_rate=1.2):
    """
    Проверяет выход за границы и корректирует позиции.
    Если ворона выходит за границы, диапазон автоматически расширяется.

    :param x: массив координат.
    :param low_b, high_b: границы по каждой координате.
    :param expand_rate: коэффициент расширения границ.

    :return: Корректированные координаты, обновленные границы.
    """
    for d in range(len(x)):
        if x[d] < low_b[d]:
            low_b[d] = x[d] * expand_rate
        elif x[d] > high_b[d]:
            high_b[d] = x[d] * expand_rate

    return np.clip(x, low_b, high_b), low_b, high_b


def initialize_bounds(problem_dimen, init_low_b, init_up_b):
    """
    Инициализация массивов нижних и верхних границ для каждой переменной.

    Если `init_low_b` и `init_up_b` — это числа или numpy‑скаляры,
    они будут применены ко всем переменным.
    Если это массивы, они будут использоваться как есть.

    :param problem_dimen: размерность задачи.
    :param init_low_b, init_up_b

    :return: low_b, high_b
    """
    if np.isscalar(init_low_b):
        low_b = np.full(problem_dimen, init_low_b, dtype=float)
    else:
        low_b = np.asarray(init_low_b, dtype=float)

    if np.isscalar(init_up_b):
        high_b = np.full(problem_dimen, init_up_b, dtype=float)
    else:
        high_b = np.asarray(init_up_b, dtype=float)

    return low_b, high_b


def initialize_positions(agents, problem_dimen, low_b, high_b, seed):
    """
    Инициализация начальных позиций.

    :param agents: количество агентов.
    :param problem_dimen: размерность задачи.
    :param low_b, up_b: начальные границы по каждой координате.
    :param seed: int (seed для генератора)

    :return Массив начальных позиций ворон.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.array([[low_b[d] + (high_b[d] - low_b[d]) * random() for d in range(problem_dimen)]
                     for _ in range(agents)])


def evaluate_fitness_serial(positions: np.ndarray, func: Any) -> np.ndarray:
    """
    Последовательное вычисление фитнеса для каждой позиции.

    :param positions
    :param func

    :return np.ndarray
    """
    return np.array([func(pos) for pos in positions])
