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

    def __init__(self, algorithm: BaseAlgorithm, iteration_info=True, seed=None):
        self.algorithm = algorithm
        self.f = iteration_info
        self.seed = seed if seed else self.algorithm.params.get('seed')

    def run(self):
        """
        Запустить алгоритм.
        Читает max_iter из algorithm.params['max_iter'].

        Возвращает:
          - result: dict (например {'minimum_x': ..., 'minimum_value': ...})
          - history_of_best_fitness: list значений best_so_far на каждой итерации
          - history_of_best_x
          - total_elapsed
        """
        # Получаем число итераций из параметров алгоритма
        max_iter = self.algorithm.max_iter
        if max_iter is None:
            raise ValueError('Parameter max_iter must be set in algorithm.params')

        # Опционально: устанавливаем seed
        if self.seed is not None:
            import numpy as np
            np.random.seed(self.seed)

        # Инициализация алгоритма
        logger.info('Initializing algorithm(%s)...', self.algorithm.__class__.__name__)
        total_start = time.perf_counter()
        self.algorithm.initialize()

        history_of_best_fitness: List[float] = []
        history_of_best_x: List[float] = []
        # Записываем значение до итераций
        try:
            current = self.algorithm.get_result()
        except Exception:
            current["minimum_value"] = None
        if current is not None:
            history_of_best_x.append(current["minimum_x"])
            history_of_best_fitness.append(current["minimum_value"])

        # Основной цикл
        for i in range(max_iter):
            iter_start = time.perf_counter()
            self.algorithm.iterate()
            elapsed_iter = time.perf_counter() - iter_start
            res = self.algorithm.get_result()
            history_of_best_fitness.append(res['minimum_value'])
            history_of_best_x.append(res["minimum_x"])
            if self.f:
                logger.info(f"Iteration {i + 1} end, time = {elapsed_iter:.4f}c,"
                            f"the best f = {res["minimum_value"]:.4e}, x = {res["minimum_x"]}")

        # Финальный результат
        total_elapsed = time.perf_counter() - total_start
        result = self.algorithm.get_result()
        if self.f:
            logger.info(f"Algorithm end: the best f = {result['minimum_value']:.4e}; "
                        f"x = {result["minimum_x"]}, time = {total_elapsed:.4f}c")

        return result, history_of_best_fitness, history_of_best_x, total_elapsed
