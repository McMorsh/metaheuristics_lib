from random import random
from typing import Dict, Any

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds, initialize_positions, enforce_boundaries_csa
from utils.mp_utils import mp_evaluate_fitness


class CrowSearchAlgorithmMP(BaseAlgorithm):
    """
    Алгоритм поиска ворона (Crow Search Algorithm, CSA) с использованием multiprocessing.Pool.

    Реализует метаэвристический алгоритм глобальной оптимизации,
    вдохновлённый стратегиями хранения и поиска пищи воронами.
    Популяция агентов («воронов») обновляет позиции в пространстве поиска
    с учётом «тайных» тайминг-памятей о найденных решениях и параметров
    вероятности осознания угрозы.

    Распараллеливание через Pool позволяет распределять вычисление фитнеса
    по нескольким рабочим процессам в рамках одного узла.

    Parameters
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
        Вероятность того, что вороны распознают и избегают спойлеров.
    expand_rate : float, default=1.2
        Коэффициент расширения границ поиска при необходимости.
    processes : int, optional
        Количество процессов в пуле для параллельного вычисления.

    Attributes
    ----------
    n_crows : int
        Число агентов в популяции.
    crows : np.ndarray, shape (agents, dim)
        Текущие позиции всех агентов.
    best_crows : np.ndarray, shape (agents, dim)
        Запомненные лучшие позиции каждого агента.
    fitness : np.ndarray, shape (agents,)
        Последние рассчитанные значения функции пригодности.
    best_fit : List[float]
        История лучших значений фитнеса на каждой итерации.
    _processes : multiprocessing.Pool
        Пул рабочих процессов для параллельного расчёта фитнеса.

    Methods
    -------
    initialize()
        Инициализация популяции, параметров Pool и распределение работы.
    iterate()
        Одна итерация обновления позиций с параллельным вычислением фитнеса.
    get_result() -> Dict[str, Any]
        Сбор и возврат глобального минимума после завершения всех итераций.
    """

    def initialize(self) -> None:
        # Сохраняем функцию оптимизации и размерность задачи
        self.fitness_function = self.problem
        self.problem_dimen = self.dim
        self._bounds = self.bounds
        self.n_crows = self.agents

        # Инициализируем генератор случайных чисел при наличии seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Параметры CSA
        self.fly_len = float(self.params.get("flight_length", 2.0))
        self.awar_prob = float(self.params.get("awareness_prob", 0.1))
        self.expand_rate = float(self.params.get('expand_rate', 1.2))
        self._processes = self.params.get('processes', None)

        # Определяем начальные границы поиска
        self._bounds = initialize_bounds(self.problem_dimen, self._bounds)

        # Инициализируем позиции воронов и запоминаем их
        self.crows = initialize_positions(self.n_crows, self.problem_dimen, self._bounds, self.seed)

        self.best_crows = self.crows.copy()

        # Всем процессам — вычислить фитнес текущих позиций
        self.fitness = mp_evaluate_fitness(self.crows, self.fitness_function, processes=self._processes)

        # Храним историю лучших значений фитнеса
        self.best_fit = []

    def iterate(self) -> None:
        """
        Одна итерация CSA обновляет позиции ворон.
        """
        # Подготовка массива для новых позиций
        new_crows = np.empty((self.n_crows, self.problem_dimen))
        num = np.random.randint(0, self.n_crows, size=self.n_crows)

        for i in range(self.n_crows):
            if random() > self.awar_prob:
                # Ворон преследует выбранную цель
                new_crows[i] = self.crows[i] + self.fly_len * (random() * (self.best_crows[num[i]] - self.crows[i]))
            else:
                # Ворон «испугался» и улетел в случайную точку
                new_crows[i] = [self._bounds[d][0] + (self._bounds[d][1] - self._bounds[d][0]) * random() for d in
                                range(self.problem_dimen)]
            # Проверяем и корректируем выход за границы
            new_crows[i], self._bounds = enforce_boundaries_csa(new_crows[i], self._bounds, self.expand_rate)

        # Параллельно вычисляем фитнес новых позиций
        new_fit = mp_evaluate_fitness(new_crows, self.fitness_function, processes=self._processes)

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
        """
        Возвращает словарь с лучшим найденным решением.
        """
        best_idx = np.argmin(self.fitness)
        best_solution = self.best_crows[best_idx]
        best_value = self.best_fit[-1] if self.best_fit else self.fitness[best_idx]
        return {"minimum_x": best_solution.tolist(), "minimum_value": float(best_value)}
