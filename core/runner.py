"""Запуск алгоритма и сбор истории сходимости."""

import time
from types import TracebackType
from typing import Self

from core.algorithm import BaseAlgorithm
from core.types import OptimizationResult
from utils.logger import get_logger

logger = get_logger(__name__)


class Runner:
    """Запускает алгоритм и собирает историю сходимости.

    Args:
        algorithm: алгоритм, готовый к инициализации.
        iteration_info: печатать ли информацию о каждой итерации.
        seed: seed для запуска. Если задан, переопределяет seed
            алгоритма.

    Для вычислителя с пулом процессов используйте ``Runner`` как
    контекстный менеджер, чтобы закрыть пул после запуска.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        iteration_info: bool = True,
        seed: int | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.iteration_info = iteration_info
        self.seed = seed

    def run(
        self,
    ) -> tuple[
        OptimizationResult,
        list[float],
        list[list[float]],
        float,
    ]:
        """Запускает алгоритм.

        Returns:
            Кортеж из результата, истории лучшего значения функции,
            истории лучших координат и затраченного времени. История
            содержит начальную точку, поэтому её длина равна
            ``max_iter + 1``.
        """
        seed = self.seed if self.seed is not None else self.algorithm.seed
        if seed is not None:
            self.algorithm.set_seed(seed)

        max_iter = self.algorithm.max_iter
        if max_iter < 1:
            raise ValueError(
                "Параметр max_iter должен быть положительным, "
                f"получено {max_iter}"
            )

        if self.algorithm.is_root:
            logger.info(
                f"Initializing algorithm({type(self.algorithm).__name__})..."
            )

        total_start = time.perf_counter()
        self.algorithm.initialize()

        result = self.algorithm.get_result()
        history_of_best_fitness = [result["minimum_value"]]
        history_of_best_x = [list(result["minimum_x"])]

        for iteration in range(max_iter):
            iteration_start = time.perf_counter()
            self.algorithm.iterate()
            iteration_elapsed = time.perf_counter() - iteration_start

            result = self.algorithm.get_result()
            history_of_best_fitness.append(result["minimum_value"])
            history_of_best_x.append(list(result["minimum_x"]))

            if self.iteration_info and self.algorithm.is_root:
                logger.info(
                    f"Iteration {iteration + 1} end, "
                    f"time = {iteration_elapsed:.4f} c, "
                    f"the best f = {result['minimum_value']:.4e}, "
                    f"x = {result['minimum_x']}"
                )

        total_elapsed = time.perf_counter() - total_start
        if self.algorithm.is_root:
            logger.info(
                f"Algorithm end: the best f = "
                f"{result['minimum_value']:.4e}; "
                f"x = {result['minimum_x']}, "
                f"time = {total_elapsed:.4f} c"
            )

        return (
            result,
            history_of_best_fitness,
            history_of_best_x,
            total_elapsed,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.algorithm.close()
