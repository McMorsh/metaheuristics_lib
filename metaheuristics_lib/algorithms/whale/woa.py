from typing import Any, Dict

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import evaluate_fitness_serial, initialize_bounds, initialize_positions, enforce_boundaries
from utils.logger import get_logger

logger = get_logger(__name__, level="INFO", log_to_file=False)


class WhaleOptimizationAlgorithm(BaseAlgorithm):
    """
    Whale Optimization Algorithm (WOA)

    Реализует метаэвристический алгоритм оптимизации,
    вдохновлённый социальным поведением горбатых китов при охоте:
    обволакивание жертвы пузырным облаком и её захват.

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

    Methods
    -------
    initialize()
        Инициализация популяции и памяти агентов.
    iterate()
        Одна итерация обновления позиций согласно правилам CSA.
    get_result() -> Dict[str, Any]
        Возвращает словарь с ключами `minimum_x` и `minimum_value`.
    """

    def initialize(self) -> None:
        # Сохраняем функцию оптимизации и размерность задачи
        self.fitness_function = self.problem
        self.problem_dimen = self.dim
        self._bounds = self.bounds
        self.n_whales = self.agents
        self._max_iter = self.max_iter

        # Инициализируем генератор случайных чисел при наличии seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Определяем начальные границы поиска
        self._bounds = initialize_bounds(self.problem_dimen, self._bounds)

        # Инициализация популяции китов
        self.whales = initialize_positions(self.n_whales, self.problem_dimen, self._bounds, self.seed)
        # print(self.whales)

        # Вычисление значений fitness для всех китов
        fitness = evaluate_fitness_serial(self.whales, self.fitness_function)

        # print(fitness)
        self.best_whale = self.whales[np.argmin(fitness)]  # Лучший кит
        self.best_fitness = float(np.min(fitness))  # Минимум функции

        self.iteration = 0

    def iterate(self) -> None:
        """
        Одна итерация WOA обновляет позиции китов.
        """
        # Линейное уменьшение параметра a от 2 до 0
        a = 2 - 2 * self.iteration / self._max_iter  # Линейное уменьшение параметра a

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

            # Ограничение позиций китов в пределах bounds
            self.whales[i] = enforce_boundaries(new_pos, self._bounds)

        # Обновление fitness
        fitness = evaluate_fitness_serial(self.whales, self.fitness_function)

        current_best = float(np.min(fitness))
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_whale = self.whales[np.argmin(fitness)]

        self.iteration += 1

    def get_result(self) -> Dict[str, Any]:
        """
        Возвращает словарь с лучшим найденным решением.
        """
        return {"minimum_x": self.best_whale.tolist(), "minimum_value": float(self.best_fitness)}
