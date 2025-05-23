from typing import Any, Dict

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds
from utils.logger import get_logger
from utils.mpi_utils import rank, mpi_evaluate_fitness

logger = get_logger(__name__)


class GreyWolfOptimizer(BaseAlgorithm):
    """
    Grey Wolf Optimizer (GWO) with mpi4py

    Метаэвристический алгоритм глобальной оптимизации,
    вдохновлённый иерархическим поведением и стратегией охоты
    серых волков (alpha, beta, delta, omega).

    Распараллеливание через MPI позволяет распределять вычисление фитнеса
    и синхронизацию позиций между процессами.

    Parameters
    ----------
    problem : Callable[[np.ndarray], float]
        Целевая функция для минимизации.
    dim : int
        Размерность пространства поиска.
    bounds : Sequence[Tuple[float, float]]
        Границы поиска для каждой размерности (low, high).
    agents : int
        Размер популяции (число волков).
    max_iter : int
        Максимальное число итераций.
    seed : int, optional
        Seed для генератора случайных чисел.

    Attributes
    ----------
    wolves : np.ndarray, shape (agents, dim)
        Позиции всех агентов (волков).
    fitness : np.ndarray, shape (agents,)
        Значения целевой функции для текущих позиций.
    alpha_pos : np.ndarray, shape (dim,)
        Позиция лидирующего волка (alpha).
    beta_pos : np.ndarray, shape (dim,)
        Позиция второго по рангу волка (beta).
    delta_pos : np.ndarray, shape (dim,)
        Позиция третьего по рангу волка (delta).
    alpha_fit, beta_fit, delta_fit : float
        Соответствующие значения функции пригодности.
    iteration : int
        Номер текущей итерации.
    comm : MPI.Comm
        Коммуникатор MPI для обмена данными между процессами.
    rank : int
        Ранг текущего процесса в MPI.
    size : int
        Общее число MPI-процессов.

    Methods
    -------
    initialize()
        Инициализация популяции и памяти агентов.
    iterate()
        Одна итерация обновления позиций согласно правилам CSA.
    get_result() -> Dict[str, Any]
        Возвращает словарь с ключами `minimum_x` и `minimum_value`.

    Примечания
    ----------
    """

    def initialize(self) -> None:
        if rank == 0:
            # Сохраняем функцию оптимизации и размерность задачи
            self.fitness_function = self.problem
            self.problem_dimen: int = self.dim
            self._bounds = np.array(self.bounds)
            self.n_wolves: int = self.agents
            self._max_iter: int = self.max_iter

            # Инициализируем генератор случайных чисел при наличии seed
            if self.seed is not None:
                np.random.seed(self.seed)

            # Определяем начальные границы поиска
            self.low_bounds, self.high_bounds = initialize_bounds(self.problem_dimen, self._bounds[0][0],
                                                                  self._bounds[0][1])

            # Инициализация популяции волков
            self.wolves = np.random.uniform(low=self.low_bounds,
                                            high=self.high_bounds,
                                            size=(self.n_wolves, self.problem_dimen))

        # Параллельная оценка фитнесса для каждого волка
        fitness = mpi_evaluate_fitness(self.wolves, self.fitness_function)

        if rank == 0:
            # Определяем лидеров: alpha, beta, delta
            idx = np.argsort(fitness)

            self.alpha_pos = self.wolves[idx[0]].copy()
            self.alpha_fit = fitness[idx[0]]

            self.beta_pos = self.wolves[idx[1]].copy()
            self.beta_fit = fitness[idx[1]]

            self.delta_pos = self.wolves[idx[2]].copy()
            self.delta_fit = fitness[idx[2]]

            self.iteration = 0

    def iterate(self) -> None:

        """
        Одна итерация CSA обновляет позиции ворон.
        """
        if rank == 0:
            # Параметр a линейно убывает от 2 до 0
            a = 2 - self.iteration * (2 / self._max_iter)

        for i in range(self.n_wolves):
            # Оценка пригодности текущего решения
            fitness = mpi_evaluate_fitness(self.wolves, self.fitness_function)

            if rank == 0:
                # Обновление alpha, beta и delta
                if fitness[i] < self.alpha_fit:
                    self.alpha_fit = fitness[i]
                    self.alpha_pos = self.wolves[i].copy()
                elif fitness[i] < self.beta_fit:
                    self.beta_fit = fitness[i]
                    self.beta_pos = self.wolves[i].copy()
                elif fitness[i] < self.delta_fit:
                    self.delta_fit = fitness[i]
                    self.delta_pos = self.wolves[i].copy()

        if rank == 0:
            # Обновление позиций всех волков
            for i in range(self.n_wolves):
                for j in range(self.problem_dimen):
                    # Коэффициенты для alpha
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.wolves[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    # Коэффициенты для beta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.wolves[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    # Коэффициенты для delta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.wolves[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # Обновляем позицию волка по среднему трех компонентов
                    self.wolves[i, j] = (X1 + X2 + X3) / 3

                # Проверка границ диапазона поиска
                self.wolves[i] = np.clip(self.wolves[i], self.low_bounds, self.high_bounds)

        self.iteration += 1

    def get_result(self) -> Dict[str, Any]:
        """
        Возвращает словарь с лучшим найденным решением.
        """
        if rank == 0:
            return {"minimum_x": self.alpha_pos.tolist(), "minimum_value": float(self.alpha_fit)}
        return None
