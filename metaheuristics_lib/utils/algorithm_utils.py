from random import random
from typing import Any

import numpy as np


def enforce_boundaries(x, low_b, up_b, expand_rate=1.2):
    """
    Проверяет выход за границы и корректирует позиции.
    Если ворона выходит за границы, диапазон автоматически расширяется.

    Параметры:
    - x: массив координат.
    - low_b, up_b: границы по каждой координате.
    - expand_rate: коэффициент расширения границ.

    Возвращает:
    - Корректированные координаты, обновленные границы.
    """
    for d in range(len(x)):
        if x[d] < low_b[d]:
            low_b[d] = x[d] * expand_rate
        elif x[d] > up_b[d]:
            up_b[d] = x[d] * expand_rate

    return np.clip(x, low_b, up_b), low_b, up_b


def initialize_bounds(problem_dimen, init_low_b, init_up_b):
    """
    Инициализация массивов нижних и верхних границ для каждой переменной.
    Если `init_low_b` и `init_up_b` — это числа или numpy‑скаляры,
    они будут применены ко всем переменным.
    Если это массивы, они будут использоваться как есть.
    """
    if np.isscalar(init_low_b):
        low_b = np.full(problem_dimen, init_low_b, dtype=float)
    else:
        low_b = np.asarray(init_low_b, dtype=float)

    if np.isscalar(init_up_b):
        up_b = np.full(problem_dimen, init_up_b, dtype=float)
    else:
        up_b = np.asarray(init_up_b, dtype=float)

    return low_b, up_b


def initialize_positions(crow_num, problem_dimen, low_b, up_b, seed):
    """
    Инициализация начальных позиций.
    Параметры:
    - crow_num: количество ворон.
    - problem_dimen: размерность задачи.
    - low_b, up_b: начальные границы по каждой координате.
    - seed: int (seed для генератора)

    Возвращает:
    - Массив начальных позиций ворон.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.array([[low_b[d] + (up_b[d] - low_b[d]) * random() for d in range(problem_dimen)]
                     for _ in range(crow_num)])


def evaluate_fitness_serial(positions: np.ndarray, func: Any) -> np.ndarray:
    """
    Последовательное вычисление фитнеса для каждой позиции.
    """
    return np.array([func(pos) for pos in positions])
