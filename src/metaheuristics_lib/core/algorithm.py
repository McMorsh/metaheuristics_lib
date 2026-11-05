"""Базовый класс метаэвристических алгоритмов."""

from abc import ABC, abstractmethod
from numbers import Real
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.evaluator import Evaluator, SerialEvaluator
from metaheuristics_lib.core.types import (
    BoundsLike,
    Objective,
    OptimizationResult,
    normalize_bounds,
)

STOP_REASON_TARGET = "target"
STOP_REASON_MAX_FEV = "max_fev"


class BaseAlgorithm(ABC):
    """Базовый класс алгоритма глобальной оптимизации.

    Класс хранит общие параметры, генератор случайных чисел, границы
    поиска и лучшее найденное решение. Наследники реализуют
    :meth:`initialize` и :meth:`iterate`.

    Атрибут ``nfev`` считает обращения к целевой функции за текущий
    запуск. Счётчик сбрасывается в :meth:`initialize`, поэтому после
    повторного запуска он не накапливается. Каждая позиция в пакете
    считается отдельным обращением, даже если алгоритм повторно
    оценивает уже известную точку.

    Args:
        problem: целевая функция, принимающая вектор и возвращающая
            значение для минимизации.
        dim: размерность пространства поиска.
        bounds: общий интервал ``(low, high)`` или интервалы по каждой
            размерности.
        agents: размер популяции.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        max_fev: мягкий лимит обращений к целевой функции. Проверяется
            на границах итераций, поэтому фактическое значение ``nfev``
            может превысить лимит в пределах одной итерации.
        target: целевое значение fitness. Если лучшее значение не больше
            ``target``, алгоритм останавливается.
        evaluator: вычислитель целевой функции. По умолчанию
            последовательный.
        **params: дополнительные параметры конкретного алгоритма.
    """

    def __init__(
        self,
        problem: Objective,
        dim: int,
        bounds: BoundsLike,
        agents: int = 100,
        max_iter: int = 50,
        seed: int | None = None,
        *,
        max_fev: int | None = None,
        target: float | None = None,
        evaluator: Evaluator | None = None,
        **params: Any,
    ) -> None:
        if not callable(problem):
            raise TypeError("Параметр problem должен быть вызываемым")
        for name, value in (
            ("dim", dim),
            ("agents", agents),
            ("max_iter", max_iter),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(
                    f"Параметр {name} должен быть целым, получено {value!r}"
                )
        if dim < 1:
            raise ValueError(
                f"Параметр dim должен быть положительным, получено {dim}"
            )
        if agents < 1:
            raise ValueError(
                f"Параметр agents должен быть положительным, получено {agents}"
            )
        if max_iter < 1:
            raise ValueError(
                "Параметр max_iter должен быть положительным, "
                f"получено {max_iter}"
            )
        if max_fev is not None:
            if isinstance(max_fev, bool) or not isinstance(
                max_fev,
                (int, np.integer),
            ):
                raise TypeError(
                    f"Параметр max_fev должен быть целым, получено {max_fev!r}"
                )
            if max_fev < 1:
                raise ValueError(
                    "Параметр max_fev должен быть положительным, "
                    f"получено {max_fev}"
                )
        if target is not None:
            if isinstance(target, bool) or not isinstance(target, Real):
                raise TypeError(
                    f"Параметр target должен быть числом, получено {target!r}"
                )
            if not np.isfinite(float(target)):
                raise ValueError(
                    "Параметр target должен быть конечным числом, "
                    f"получено {target}"
                )

        self.problem = problem
        self.dim = int(dim)
        self.bounds = normalize_bounds(bounds, self.dim)
        self.agents = int(agents)
        self.max_iter = int(max_iter)
        self.max_fev = int(max_fev) if max_fev is not None else None
        self.target = float(target) if target is not None else None
        self.seed = seed
        self.params: dict[str, Any] = dict(params)
        self.rng = np.random.default_rng(seed)
        self.evaluator: Evaluator = (
            evaluator if evaluator is not None else SerialEvaluator(problem)
        )
        self.iteration = 0
        self.nfev = 0
        self.stop_reason: str | None = None
        self.best_solution: npt.NDArray[np.float64] | None = None
        self.best_fitness = float("inf")

    @property
    def is_root(self) -> bool:
        """Возвращает ``True`` для rank 0 или последовательного режима."""
        rank = getattr(self.evaluator, "rank", 0)
        return bool(rank == 0)

    def set_seed(self, seed: int | None) -> None:
        """Сбрасывает генератор случайных чисел.

        Args:
            seed: новый seed или ``None`` для недетерминированного
                запуска.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def check_stop(self) -> str | None:
        """Проверяет условия ранней остановки.

        Returns:
            ``"target"``, если достигнуто целевое значение, ``"max_fev"``,
            если исчерпан лимит вычислений, или ``None``, если нужно
            продолжать.
        """
        if self.stop_reason is not None:
            return self.stop_reason
        if self.target is not None and self.best_fitness <= self.target:
            self.stop_reason = STOP_REASON_TARGET
        elif self.max_fev is not None and self.nfev >= self.max_fev:
            self.stop_reason = STOP_REASON_MAX_FEV
        return self.stop_reason

    def _reset_state(self) -> None:
        """Сбрасывает результаты предыдущего запуска."""
        self.iteration = 0
        self.nfev = 0
        self.stop_reason = None
        self.best_solution = None
        self.best_fitness = float("inf")

    def _evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Вычисляет fitness и увеличивает счётчик обращений.

        Args:
            positions: позиции формы ``(n, dim)``.

        Returns:
            Значения целевой функции формы ``(n,)``.

        Raises:
            ValueError: если fitness содержит ``NaN`` или бесконечность.
        """
        values = self.evaluator.evaluate(positions)
        self.nfev += int(positions.shape[0])
        self._validate_fitness(values)
        return values

    @staticmethod
    def _validate_fitness(
        fitness: npt.NDArray[np.float64],
    ) -> None:
        """Проверяет, что все значения fitness конечны.

        Args:
            fitness: значения целевой функции.

        Raises:
            ValueError: если среди значений есть ``NaN`` или
                бесконечность.
        """
        invalid = ~np.isfinite(fitness)
        if not np.any(invalid):
            return

        index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            "Значения fitness должны быть конечными, "
            f"получено {fitness[index]} для позиции {index}"
        )

    def _initial_positions(self) -> npt.NDArray[np.float64]:
        """Генерирует начальную популяцию внутри границ."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return self.rng.uniform(
            low=low,
            high=high,
            size=(self.agents, self.dim),
        )

    def _clip(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Ограничивает позиции границами поиска."""
        return np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])

    def _update_best(
        self,
        positions: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
    ) -> None:
        """Обновляет лучшее решение, если найдено улучшение.

        Args:
            positions: позиции агентов формы ``(n, dim)``.
            fitness: значения функции для этих позиций.

        Raises:
            ValueError: если число позиций и значений не совпадает
                или fitness содержит ``NaN`` или бесконечность.
        """
        if positions.ndim != 2:
            raise ValueError(
                "Позиции должны быть двумерным массивом, "
                f"получена форма {positions.shape}"
            )
        if positions.shape[0] != fitness.shape[0]:
            raise ValueError(
                "Число позиций и значений fitness должно совпадать, "
                f"получено {positions.shape[0]} и {fitness.shape[0]}"
            )
        self._validate_fitness(fitness)

        index = int(np.argmin(fitness))
        value = float(fitness[index])
        if value < self.best_fitness:
            self.best_fitness = value
            self.best_solution = np.array(
                positions[index],
                dtype=np.float64,
                copy=True,
            )

    def _best_position(self) -> npt.NDArray[np.float64]:
        """Возвращает лучшее решение или поднимает ошибку."""
        if self.best_solution is None:
            raise RuntimeError(
                "Алгоритм не инициализирован: лучшее решение отсутствует"
            )
        return self.best_solution

    def get_result(self) -> OptimizationResult:
        """Возвращает лучшее найденное решение.

        Returns:
            Словарь с координатами и значением целевой функции.

        Raises:
            RuntimeError: если алгоритм ещё не инициализирован.
        """
        best = self._best_position()
        return OptimizationResult(
            minimum_x=[float(value) for value in best],
            minimum_value=float(self.best_fitness),
        )

    def close(self) -> None:
        """Освобождает ресурсы вычислителя."""
        self.evaluator.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dim={self.dim}, agents={self.agents}, "
            f"max_iter={self.max_iter})"
        )

    @abstractmethod
    def initialize(self) -> None:
        """Подготавливает популяцию и начальное лучшее решение."""

    @abstractmethod
    def iterate(self) -> None:
        """Выполняет одну итерацию алгоритма."""
