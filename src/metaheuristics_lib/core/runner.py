"""Запуск алгоритма и сбор истории сходимости."""

import logging
import time
from collections.abc import Iterable, Mapping, Set
from numbers import Real
from types import TracebackType
from typing import Self

from metaheuristics_lib.core.algorithm import BaseAlgorithm
from metaheuristics_lib.core.types import OptimizationResult

logger = logging.getLogger(__name__)


def _validate_result(
    result: object,
    expected_dim: int,
) -> OptimizationResult:
    """Проверяет и нормализует результат алгоритма.

    Args:
        result: значение, возвращённое ``get_result()``.
        expected_dim: ожидаемая длина вектора решения.

    Returns:
        Нормализованный результат.

    Raises:
        TypeError: если результат или его поля имеют неверный тип.
        KeyError: если в результате отсутствуют обязательные ключи.
        ValueError: если длина вектора не совпадает с размерностью.
    """
    if not isinstance(result, dict):
        raise TypeError(
            "get_result() должен возвращать словарь, "
            f"получено {type(result).__name__}"
        )

    required = {"minimum_x", "minimum_value"}
    missing = required.difference(result)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"В результате отсутствуют ключи: {names}")

    value = result["minimum_value"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "minimum_value должен быть числом, "
            f"получено {type(value).__name__}"
        )

    raw_x = result["minimum_x"]
    if isinstance(
        raw_x,
        (str, bytes, Mapping, Set),
    ) or not isinstance(raw_x, Iterable):
        raise TypeError(
            "minimum_x должен быть последовательностью чисел, "
            f"получено {type(raw_x).__name__}"
        )

    coordinates: list[float] = []
    for coordinate in raw_x:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise TypeError("minimum_x должен содержать только числа")
        coordinates.append(float(coordinate))

    if len(coordinates) != expected_dim:
        raise ValueError(
            f"minimum_x должен содержать {expected_dim} координат, "
            f"получено {len(coordinates)}"
        )

    return OptimizationResult(
        minimum_x=coordinates,
        minimum_value=float(value),
    )


class Runner:
    """Запускает алгоритм и собирает историю сходимости.

    Args:
        algorithm: алгоритм, готовый к инициализации.
        iteration_info: ``False`` отключает лог итераций, ``True``
            логирует каждую итерацию, целое число задаёт интервал.
        seed: seed для запуска. Если задан, переопределяет seed
            алгоритма.

    Без контекстного менеджера ресурсы закрываются после завершения
    ``run``, поэтому повторные запуски следует выполнять внутри
    ``with Runner(...)``, особенно для пула процессов.

    Raises:
        TypeError: если ``iteration_info`` не bool и не int.
        ValueError: если целое ``iteration_info`` меньше 1.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        iteration_info: bool | int = True,
        seed: int | None = None,
    ) -> None:
        self.algorithm = algorithm
        self._iteration_interval: int | None
        if isinstance(iteration_info, bool):
            self._iteration_interval = 1 if iteration_info else None
        elif isinstance(iteration_info, int):
            if iteration_info < 1:
                raise ValueError(
                    "Параметр iteration_info должен быть положительным, "
                    f"получено {iteration_info}"
                )
            self._iteration_interval = iteration_info
        else:
            raise TypeError(
                "Параметр iteration_info должен быть bool или int, "
                f"получено {type(iteration_info).__name__}"
            )
        self.iteration_info: bool | int = iteration_info
        self.seed = seed
        self._entered = False

    def run(
        self,
    ) -> tuple[
        OptimizationResult,
        list[float],
        list[list[float]],
        float,
    ]:
        """Запускает алгоритм.

        Без контекстного менеджера закрывает ресурсы после завершения.
        Для повторных запусков используйте ``with Runner(...)``.

        Returns:
            Кортеж из результата, истории лучшего значения функции,
            истории лучших координат и затраченного времени. Причина
            ранней остановки доступна в ``algorithm.stop_reason``.

        Raises:
            TypeError: если результат имеет неверный тип.
            KeyError: если в результате отсутствуют обязательные ключи.
            ValueError: если результат содержит неверное число координат.
        """
        try:
            return self._run()
        finally:
            if not self._entered:
                self.algorithm.close()

    def _run(
        self,
    ) -> tuple[
        OptimizationResult,
        list[float],
        list[list[float]],
        float,
    ]:
        """Выполняет один запуск алгоритма.

        История содержит начальную точку. Без ранней остановки её
        длина равна ``max_iter + 1``. Если алгоритм остановился до
        последней итерации, история короче.
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
        self.algorithm.check_stop()

        result = _validate_result(
            self.algorithm.get_result(),
            self.algorithm.dim,
        )
        history_of_best_fitness = [result["minimum_value"]]
        history_of_best_x = [list(result["minimum_x"])]

        for iteration in range(max_iter):
            if self.algorithm.stop_reason is not None:
                break

            iteration_start = time.perf_counter()
            self.algorithm.iterate()
            iteration_elapsed = time.perf_counter() - iteration_start

            result = _validate_result(
                self.algorithm.get_result(),
                self.algorithm.dim,
            )
            history_of_best_fitness.append(result["minimum_value"])
            history_of_best_x.append(list(result["minimum_x"]))

            if (
                self._iteration_interval is not None
                and self.algorithm.is_root
                and (iteration + 1) % self._iteration_interval == 0
            ):
                logger.info(
                    f"Iteration {iteration + 1} end, "
                    f"time = {iteration_elapsed:.4f} c, "
                    f"the best f = {result['minimum_value']:.4e}, "
                    f"x = {result['minimum_x']}"
                )

            self.algorithm.check_stop()

        total_elapsed = time.perf_counter() - total_start
        if self.algorithm.is_root:
            logger.info(
                f"Algorithm end: the best f = "
                f"{result['minimum_value']:.4e}; "
                f"x = {result['minimum_x']}, "
                f"time = {total_elapsed:.4f} c, "
                f"stop = {self.algorithm.stop_reason or 'max_iter'}"
            )

        return (
            result,
            history_of_best_fitness,
            history_of_best_x,
            total_elapsed,
        )

    def __enter__(self) -> Self:
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.algorithm.close()
        finally:
            self._entered = False
