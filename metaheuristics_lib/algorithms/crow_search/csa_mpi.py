from random import randint, random
from typing import Dict, Any

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds, initialize_positions, enforce_boundaries_csa
from utils.mpi_utils import rank, comm, mpi_evaluate_fitness


class CrowSearchAlgorithmMPI(BaseAlgorithm):
    """
    Алгоритм поиска ворона (Crow Search Algorithm, CSA) with mpi4py.

    Реализует метаэвристический алгоритм глобальной оптимизации,
    вдохновлённый стратегиями хранения и поиска пищи воронами.
    Популяция агентов («воронов») обновляет позиции в пространстве поиска
    с учётом «тайных» тайминг-памятей о найденных решениях и параметров
    вероятности осознания угрозы.

    Распараллеливание через MPI позволяет распределять вычисление фитнеса
    и синхронизацию позиций между процессами.

    Параметры
    ----------
    problem : Callable[[np.ndarray], float]
        Целевая функция, которую необходимо минимизировать.
    dim : int
        Размерность пространства поиска.
    bounds : Sequence[Tuple[float, float]]
        Список кортежей (low, high) с границами для каждой размерности.
    agents : int
        Число «воронов» (размер популяции).
    max_iter : int
        Максимальное число итераций алгоритма.
    seed : int, optional
        Seed для инициализации генератора случайных чисел.
    flight_length : float, default=2.0
        Коэффициент длины полёта (step size) при обновлении позиций.
    awareness_prob : float, default=0.1
        Вероятность того, что вороны распознают и избегают «спойлеров» (awareness).
    expand_rate : float, default=1.2
        Коэффициент расширения границ поиска при необходимости.

    Атрибуты
    --------
    n_crows : np.ndarray, shape (agents, dim)
        Текущие позиции всех агентов.
    best_crows : np.ndarray, shape (agents, dim)
        Лучшие (запомненные) позиции каждого агента.
    fitness : np.ndarray, shape (agents,)
        Текущие значения функции пригодности для каждой позиции.
    best_fit : float
        Лучшая приспособленность на каждой итерации.
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

    Примечания
    ----------
    """

    def initialize(self) -> None:
        # Только главный процесс задаёт исходные данные
        if rank == 0:
            # Сохраняем функцию оптимизации и размерность задачи
            self.fitness_function = self.problem
            self.problem_dimen = self.dim
            self._bounds = np.array(self.bounds)
            self.n_crows = self.agents

            # Инициализируем генератор случайных чисел при наличии seed
            if self.seed is not None:
                np.random.seed(self.seed)

            # Параметры CSA
            self.fly_len = float(self.params.get("flight_length", 2.0))
            self.awar_prob = float(self.params.get("awareness_prob", 0.1))
            self.expand_rate = float(self.params.get('expand_rate', 1.2))

            # Определяем начальные границы поиска
            self.low_bounds, self.high_bounds = initialize_bounds(self.problem_dimen,
                                                                  self._bounds[0][0], self._bounds[0][1])

            # Инициализируем позиции воронов и запоминаем их
            self.crows = initialize_positions(self.n_crows, self.problem_dimen,
                                              self.low_bounds, self.high_bounds, self.seed)
            self.best_crows = self.crows.copy()

            # Храним историю лучших значений фитнеса
            self.best_fit = []

        # Всем процессам — вычислить фитнес текущих позиций
        self.fitness = mpi_evaluate_fitness(self.crows, self.fitness_function)

    def iterate(self) -> None:
        """
        Одна итерация CSA обновляет позиции ворон.
        """
        # Генерация новых позиций выполняется только на rank 0
        if rank == 0:
            # Подготовка массива для новых позиций
            new_crows = np.empty((self.n_crows, self.problem_dimen))
            num = np.array([randint(0, self.n_crows - 1) for _ in range(self.n_crows)])

            for i in range(self.n_crows):
                if random() > self.awar_prob:
                    # Ворон преследует выбранную цель
                    new_crows[i] = self.crows[i] + self.fly_len * (random() * (self.best_crows[num[i]] - self.crows[i]))
                else:
                    # Ворон «испугался» и улетел в случайную точку
                    new_crows[i] = [self.low_bounds[d] + (self.high_bounds[d] - self.low_bounds[d]) * random()
                                    for d in range(self.problem_dimen)]
                # Проверяем и корректируем выход за границы
                new_crows[i], self.low_bounds, self.high_bounds = enforce_boundaries_csa(new_crows[i],
                                                                                         self.low_bounds,
                                                                                         self.high_bounds,
                                                                                         self.expand_rate)
        else:
            new_crows = None

        # Распространяем новые позиции на все процессы
        new_crows = comm.bcast(new_crows, root=0)

        # Вычисляем фитнес новых позиций через MPI
        if rank == 0:
            new_fit = mpi_evaluate_fitness(new_crows, self.fitness_function)
        else:
            # Непосредственный вызов внутри рабочих процессов
            mpi_evaluate_fitness(new_crows, self.fitness_function)
            new_fit = None

        # Синхронизируем результаты фитнеса
        new_fit = comm.bcast(new_fit, root=0)

        # Только главный процесс обновляет память и текущие позиции
        if rank == 0:
            # Если заданы дополнительные границы на значение фитнеса — применяем
            if self._bounds.shape[0] > 1:
                y_min, y_max = self._bounds[1]
                for i in range(self.n_crows):
                    if ((y_min is not None and new_fit[i] < y_min) or
                            (y_max is not None and new_fit[i] > y_max)):
                        new_fit[i] = np.inf

            # Обновляем память для каждого ворона
            for i in range(self.n_crows):
                if new_fit[i] < self.fitness[i]:
                    self.best_crows[i] = new_crows[i].copy()
                    self.fitness[i] = new_fit[i]

            # Сохраняем историческое лучшее значение за итерацию
            self.best_fit.append(np.min(self.fitness))
            # Обновляем текущие позиции
            self.crows = new_crows.copy()

    def get_result(self) -> Dict[str, Any]:
        if rank == 0:
            best_idx = np.argmin(self.fitness)
            best_solution = self.best_crows[best_idx]
            best_value = self.best_fit[-1] if self.best_fit else self.fitness[best_idx]
            return {"minimum_x": best_solution.tolist(), "minimum_value": float(best_value)}
        return None
