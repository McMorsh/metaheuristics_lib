# algorithms/mp_utils.py
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np


def mp_evaluate_fitness(positions: np.ndarray, func: Callable[[np.ndarray], float],
                        processes: int = None) -> np.ndarray:
    """
    Параллельное вычисление фитнеса
    Каждая позиция – это np.ndarray, возвращается np.ndarray скалярных значений.
    """
    if processes is None:
        processes = os.cpu_count() // 2

    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = list(executor.map(func, positions))
    return np.array(results)