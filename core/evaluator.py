"""Вычисление целевой функции для набора позиций."""

import os
from concurrent.futures import ProcessPoolExecutor
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt

from core.types import Objective


class Evaluator(Protocol):
    """Интерфейс вычислителя целевой функции."""

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Возвращает значения функции для каждой позиции."""

    def close(self) -> None:
        """Освобождает ресурсы вычислителя."""


def _as_positions(
    positions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Проверяет, что позиции образуют двумерный массив."""
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "Позиции должны быть двумерным массивом, "
            f"получена форма {array.shape}"
        )
    return array


class SerialEvaluator:
    """Последовательный вычислитель."""

    def __init__(self, objective: Objective) -> None:
        self._objective = objective

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Вычисляет значения функции в одном процессе."""
        array = _as_positions(positions)
        return np.asarray(
            [self._objective(position) for position in array],
            dtype=np.float64,
        )

    def close(self) -> None:
        """Освобождает ресурсы; для последовательного режима нечего делать."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ProcessPoolEvaluator:
    """Вычислитель на пуле процессов.

    Пул создаётся лениво при первом вызове и переиспользуется до
    :meth:`close`.
    """

    def __init__(
        self,
        objective: Objective,
        max_workers: int | None = None,
        chunksize: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError(
                "Параметр max_workers должен быть положительным, "
                f"получено {max_workers}"
            )
        if chunksize is not None and chunksize < 1:
            raise ValueError(
                "Параметр chunksize должен быть положительным, "
                f"получено {chunksize}"
            )
        self._objective = objective
        self._max_workers = max_workers
        self._chunksize = chunksize
        self._executor: ProcessPoolExecutor | None = None

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Вычисляет значения функции в пуле процессов."""
        array = _as_positions(positions)
        if array.size == 0:
            return np.empty(0, dtype=np.float64)

        workers = self._max_workers or os.cpu_count() or 1
        if workers == 1:
            return np.asarray(
                [self._objective(position) for position in array],
                dtype=np.float64,
            )

        executor = self._get_executor()
        chunksize = self._chunksize or max(1, len(array) // (workers * 4))
        objective = self._objective
        values = list(executor.map(objective, array, chunksize=chunksize))
        return np.asarray(values, dtype=np.float64)

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        return self._executor

    def close(self) -> None:
        """Останавливает пул процессов."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class MPIEvaluator:
    """Вычислитель, распределяющий позиции между рангами MPI.

    Все ранги получают полный массив значений функции: rank 0 собирает
    результаты и рассылает их через ``bcast``. Благодаря этому состояние
    алгоритма остаётся одинаковым на всех рангах.
    """

    def __init__(self, objective: Objective) -> None:
        try:
            from mpi4py import MPI
        except ImportError as error:
            raise RuntimeError(
                "Для MPI-режима установите mpi4py: "
                "pip install 'metaheuristics-lib[mpi]'"
            ) from error

        self._comm: Any = MPI.COMM_WORLD
        self.rank = int(self._comm.Get_rank())
        self.size = int(self._comm.Get_size())
        self._objective = objective

    def common_seed(self, seed: int | None) -> int:
        """Возвращает одинаковый seed на всех рангах."""
        value = seed
        if self.rank == 0 and value is None:
            value = int(np.random.default_rng().integers(0, 2**32 - 1))
        return int(self._comm.bcast(value, root=0))

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Распределяет позиции по рангам и собирает значения функции."""
        array = _as_positions(positions)
        chunks = np.array_split(array, self.size) if self.rank == 0 else None
        local_positions = np.asarray(
            self._comm.scatter(chunks, root=0),
            dtype=np.float64,
        )

        # Ошибку нужно разослать всем рангам, иначе остальные зависнут
        # в gather или bcast. Поэтому перехватываем Exception и
        # передаём текст ошибки через allgather.
        error: str | None = None
        try:
            local_fitness = np.asarray(
                [self._objective(position) for position in local_positions],
                dtype=np.float64,
            )
        except Exception as error_value:
            error = f"{type(error_value).__name__}: {error_value}"
            local_fitness = np.empty(0, dtype=np.float64)

        errors: list[str | None] = self._comm.allgather(error)
        first_error = next(
            (message for message in errors if message is not None),
            None,
        )
        if first_error is not None:
            raise RuntimeError(
                f"Ошибка вычисления fitness на MPI-ранге: {first_error}"
            )

        gathered = self._comm.gather(local_fitness, root=0)
        fitness = np.concatenate(gathered) if self.rank == 0 else None
        return np.asarray(self._comm.bcast(fitness, root=0), dtype=np.float64)

    def close(self) -> None:
        """Освобождает ресурсы; коммуникатор MPI закрывать не нужно."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
