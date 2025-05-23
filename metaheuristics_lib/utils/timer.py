'''
utils/timer.py

Модуль предоставляет простые декораторы и контекстные менеджеры
для измерения времени выполнения функций и блоков кода.
Используется для профилирования алгоритмов и сбора статистики времени.
'''
import time
from functools import wraps
from typing import Callable, Tuple, Any, Optional


def timed(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.

    Возвращает кортеж (result, elapsed_time).
    Можно применять к любым функциям: алгоритмам, вспомогательным процедурам и т.п.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        return result, elapsed

    return wrapper


class Timer:
    """
    Контекстный менеджер для измерения времени блока кода.

    Пример:
        with Timer() as t:
            fun()
        print(f"Elapsed: {t.elapsed:.4f}s")
    """

    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> 'Timer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start  # type: ignore
