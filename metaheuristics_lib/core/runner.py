import time
from typing import List, Any, Dict

from core.algorithm import BaseAlgorithm
from utils.logger import get_logger

logger = get_logger(__name__, level="INFO", log_to_file=False)


class Runner:
    """
    Runner запускает алгоритм и собирает историю сходимости.

    Параметры берутся из algo.params (max_iter, seed и др.),
    поэтому при создании алгоритма задавать все через его конструктор.
    """

    def __init__(self, algorithm: BaseAlgorithm, iteration_info=True):
        self.algorithm = algorithm
        self.f = iteration_info

    def run(self) -> (Dict[str, Any], List[float]):
        """
        Запустить алгоритм.
        Читает max_iter из algorithm.params['max_iter'].

        Возвращает:
          - result: dict (например {'minimum_x': ..., 'minimum_value': ...})
          - history: list значений best_so_far на каждой итерации
        """
        # Получаем число итераций из параметров алгоритма
        max_iter = self.algorithm.max_iter
        if max_iter is None:
            raise ValueError('Parameter max_iter must be set in algorithm.params')

        # Опционально: устанавливаем seed
        seed = self.algorithm.params.get('seed')
        if seed is not None:
            import numpy as np
            np.random.seed(seed)

        # Инициализация алгоритма
        logger.info('Initializing algorithm(%s)...', self.algorithm.__class__.__name__)
        total_start = time.perf_counter()
        self.algorithm.initialize()

        history: List[float] = []
        # Записываем значение до итераций
        try:
            current = self.algorithm.get_result()['minimum_value']
        except Exception:
            current = None
        if current is not None:
            history.append(current)

        # Основной цикл
        for i in range(max_iter):
            iter_start = time.perf_counter()
            # logger.info(f'Running iteration {i + 1}/{max_iter}...')
            self.algorithm.iterate()
            elapsed_iter = time.perf_counter() - iter_start
            res = self.algorithm.get_result()
            history.append(res['minimum_value'])
            if self.f:
                logger.info(f"Iteration {i + 1} end, time = {elapsed_iter:.4f}c,"
                            f" "f"the best f = {current:.4e}")

        # Финальный результат
        total_elapsed = time.perf_counter() - total_start
        result = self.algorithm.get_result()
        logger.info(f"Algorithm end: the best f = {result['minimum_value']:.4e}; "
                    f"time = {total_elapsed:.4f}c")
        return result, history
