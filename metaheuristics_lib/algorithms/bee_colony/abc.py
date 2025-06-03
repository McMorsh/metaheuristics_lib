from typing import Dict, Any

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import evaluate_fitness_serial, initialize_bounds, initialize_positions, enforce_boundaries


class ArtificialBeeColony(BaseAlgorithm):
    """
    Artificial Bee Colony

    Метаэвристический алгоритм оптимизации, вдохновлённый поведением
    рабочих, наблюдателей и скаутов в улье.

    Основные фазы:
      1. Фаза рабочих пчел (employed bees)
      2. Фаза наблюдающих пчел (onlooker bees)
      3. Фаза скаутов (scout bees)

    Parameters
    ----------
    problem : Callable[[np.ndarray], float]
        Функция-цель для минимизации.
    dim : int
        Размерность пространства поиска.
    bounds : Sequence[Tuple[float, float]]
        Границы поиска для каждой размерности (low, high).
    agents : int
        Число пчёл в популяции.
    max_iter : int
        Максимальное число итераций.
    seed : int, optional
        Seed для генератора случайных чисел (для воспроизводимости).
    limit : int, optional
        Число итераций без улучшения до превращения в скаута.

    Attributes
    ----------
    bees : np.ndarray, shape (agents, dim)
        Текущие позиции всех пчёл.
    fitness : np.ndarray, shape (agents,)
        Текущие значения функции пригодности.
    best_bee : np.ndarray, shape (agents, dim)
        Локальные лучшие позиции для каждой пчелы.
    best_fitness : np.ndarray, shape (agents,)
        Локальные лучшие значения fitness.
    trial : np.ndarray, shape (agents,)
        Счётчик безуспешных попыток для каждой пчелы.
    global_best_bee : np.ndarray, shape (dim,)
        Глобально лучшее найденное решение.
    global_best_fitness : float
        Значение fitness в global_best_bee.
    iteration : int
        Номер текущей итерации.

    Methods
    -------
    initialize()
        Инициализация популяции и памяти агентов.
    iterate()
        Одна итерация обновления позиций согласно правилам ABC.
    get_result() -> Dict[str, Any]
        Возвращает словарь с ключами `minimum_x` и `minimum_value`.

    Примечания
    ----------
    """

    def initialize(self) -> None:
        # Сохраняем функцию оптимизации и размерность задачи
        self.fitness_function = self.problem
        self.problem_dimen = self.dim
        self._bounds = self.bounds
        self.n_bees = self.agents
        self._max_iter = self.max_iter

        # Инициализируем генератор случайных чисел при наличии seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Параметры ABC
        self._limit = int(self.params.get("limit", 25))

        # Определяем начальные границы поиска
        self._bounds = initialize_bounds(self.problem_dimen, self._bounds)

        # Инициализируем случайные позиции пчёл
        self.bees = initialize_positions(self.n_bees, self.problem_dimen, self._bounds, self.seed)

        # Вычисление значений fitness для всех пчел
        self.fitness = evaluate_fitness_serial(self.bees, self.fitness_function)

        # Локальные лучшие и счётчики проб
        self.best_bee = self.bees.copy()
        self.best_fitness = self.fitness.copy()
        # Счётчик неудачных попыток
        self.trial = np.zeros(self.n_bees, dtype=int)

        # Глобальный лучший
        idx = np.argmin(self.best_fitness)
        self.global_best_bee = self.best_bee[idx]
        self.global_best_fitness = float(self.best_fitness[idx])

        self.iteration = 0

    def _neighbour(self, i):
        """
        Генерация соседнего решения для заданного источника
        """
        # Выбираем случайного соседа k != i
        k = np.random.choice([j for j in range(self.n_bees) if j != i])
        phi = np.random.uniform(-1, 1, self.problem_dimen)

        # Генерируем новое решение
        new_solution = self.bees[i] + phi * (self.bees[i] - self.bees[k])
        new_solution = enforce_boundaries(new_solution, self._bounds)

        fit = self.fitness_function(new_solution)

        # Принимаем, если улучшение или увеличиваем trial
        if fit < self.best_fitness[i]:
            self.bees[i] = new_solution
            self.fitness[i] = fit
            self.best_bee[i] = new_solution
            self.best_fitness[i] = fit
            self.trial[i] = 0
        else:
            self.trial[i] += 1

    def iterate(self) -> None:
        """
        Одна итерация ABC обновляет позиции пчел.
        """
        # Этап рабочих пчел
        for _i in range(self.n_bees):
            # Выбираем другой источник пищи
            self._neighbour(_i)

        # Этап наблюдающих пчел (выбор источника по вероятностям)
        # Рассчитываем вероятности на основе fitness
        probs = (1.0 / (1.0 + abs(self.best_fitness)))
        probs = probs / probs.sum()
        for _ in range(self.n_bees):
            __i = np.random.choice(self.n_bees, p=probs)
            self._neighbour(__i)

        # Фаза скаутов: заменяем источники с trial > limit
        for i in range(self.n_bees):
            if self.trial[i] > self._limit:
                # создаём случайный источник
                new_bees = initialize_positions(self.problem_dimen, self.problem_dimen, self._bounds, self.seed)[0]
                new_fitness = self.fitness_function(new_bees)
                self.bees[i] = new_bees
                self.fitness[i] = new_fitness
                self.best_bee[i] = new_bees.copy()
                self.best_fitness[i] = new_fitness
                self.trial[i] = 0

        # Обновляем глобальное лучшее решение
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.global_best_fitness:
            self.global_best_bee = self.bees[idx]
            self.global_best_fitness = self.fitness[idx]

        self.iteration += 1

    def get_result(self) -> Dict[str, Any]:
        """
        Возвращает словарь с лучшим найденным решением.
        """
        return {"minimum_x": self.global_best_bee.tolist(), "minimum_value": float(self.global_best_fitness)}
