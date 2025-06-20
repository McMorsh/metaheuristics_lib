from typing import Any, Dict

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds
from utils.logger import get_logger
from utils.mpi_utils import mpi_evaluate_fitness, rank

logger = get_logger(__name__, level="INFO", log_to_file=False)


class WhaleOptimizationAlgorithmMPI(BaseAlgorithm):
    """
    Whale Optimization Algorithm (WOA) with mpi4py

    Реализует метаэвристический алгоритм оптимизации,
    вдохновлённый социальным поведением горбатых китов при охоте:
    обволакивание жертвы пузырным облаком и её захват.

    Распараллеливание через MPI позволяет распределять вычисление фитнеса
    и синхронизацию позиций между процессами.

    Parameters
    ----------
    problem : Callable[[np.ndarray], float]
        Целевая функция для минимизации.
    dim : int
        Размерность пространства поиска.
    bounds : Sequence[Tuple[float, float]]
        Границы поиска для каждой размерности в формате (low, high).
    agents : int
        Количество агентов (китов) в популяции.
    max_iter : int
        Максимальное число итераций.
    seed : int, optional
        Seed для генератора случайных чисел.

    Attributes
    ----------
    whales : np.ndarray, shape (agents, dim)
        Текущие позиции агентов.
    fitness : np.ndarray, shape (agents,)
        Значения целевой функции для каждой позиции.
    best_whale : np.ndarray, shape (dim,)
        Лучшая найденная позиция (минимум).
    best_fitness : float
        Значение функции в best_whale.
    iteration : int
        Счетчик итераций.
    comm : MPI.Comm
        Коммуникатор MPI для обмена данными между процессами.
    rank : int
        Ранг текущего процесса в MPI.
    size : int
        Общее число MPI-процессов.

    Методы
    -------
    initialize()
        Инициализация популяции и памяти агентов.
    iterate()
        Одна итерация обновления позиций согласно правилам CSA.
    get_result() -> Dict[str, Any]
        Возвращает словарь с ключами `minimum_x` и `minimum_value`.
    """

    def initialize(self) -> None:
        # аналогично WOA, но инициализация только на rank 0
        if rank == 0:
            # Сохраняем функцию оптимизации и размерность задачи
            self.fitness_function = self.problem
            self.problem_dimen = self.dim
            self._bounds = np.array(self.bounds)
            self.n_whales = self.agents
            self._max_iter = self.max_iter

            # Инициализируем генератор случайных чисел при наличии seed
            if self.seed is not None:
                np.random.seed(self.seed)

            # Определяем начальные границы поиска
            self.low_bounds, self.high_bounds = initialize_bounds(self.problem_dimen, self._bounds[0][0],
                                                                  self._bounds[0][1])
            # Инициализация популяции китов
            self.whales = np.random.uniform(low=self.low_bounds,
                                            high=self.high_bounds,
                                            size=(self.n_whales, self.problem_dimen))

        # первый scatter/gather вызов
        fitness = mpi_evaluate_fitness(self.whales, self.fitness_function)

        if rank == 0:
            self.best_whale = self.whales[np.argmin(fitness)].copy()  # Лучший кит
            self.best_fitness = float(np.min(fitness))  # Минимум функции
            self.iteration = 0

    def iterate(self) -> None:
        # rank 0 обновляет популяцию
        if rank == 0:
            a = 2 - 2 * self.iteration / self._max_iter
            for i in range(self.n_whales):
                A = 2 * a * np.random.rand() - a  # коэфф. сжатия/разжатия
                C = 2 * np.random.rand()  # коэфф. перемещения

                if np.random.rand() < 0.5:
                    # Обновление позиции по модели сжатия
                    D = abs(C * self.best_whale - self.whales[i])
                    new_pos = self.best_whale - A * D
                else:
                    # Спиральное обновление
                    D = abs(self.best_whale - self.whales[i])
                    l = np.random.uniform(-1, 1)  # Коэффициент логарифмической спирали
                    new_pos = D * np.exp(l) * np.cos(2 * np.pi * l) + self.best_whale

                self.low_bounds, self.high_bounds = initialize_bounds(self.problem_dimen,
                                                                      self._bounds[0][0], self._bounds[0][1])
                self.whales[i] = np.clip(new_pos, self.low_bounds, self.high_bounds)

        # оценка фитнеса всеми процессами
        fitness = mpi_evaluate_fitness(self.whales, self.fitness_function)

        if rank == 0:
            # Обновление fitness
            current_best = float(np.min(fitness))
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_whale = self.whales[np.argmin(fitness)].copy()

        self.iteration += 1

    def get_result(self) -> Dict[str, Any]:
        if rank == 0:
            return {"minimum_x": self.best_whale.tolist(), "minimum_value": float(self.best_fitness)}
        return None
