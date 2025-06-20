from typing import Dict, Any

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds, initialize_positions, enforce_boundaries
from utils.mp_utils import mp_evaluate_fitness


class EmperorPenguinOptimizerMP(BaseAlgorithm):
    """
    Emperor Penguin Optimizer с использованием multiprocessing.Pool

    Реализует метаэвристический алгоритм глобальной оптимизации,
    вдохновлённый социальным поведением императорских пингвинов
    при совместном поиске пищи и сохранении тепла.

    Распараллеливание через Pool позволяет распределять вычисление фитнеса
    по нескольким рабочим процессам в рамках одного узла.

    Parameters
    ----------
    problem : Callable[[np.ndarray], float]
        Целевая функция, которую необходимо минимизировать.
    dim : int
        Размерность пространства поиска.
    bounds : Sequence[Tuple[float, float]]
        Список (low, high)-кортежей с границами для каждой размерности.
    agents : int
        Число агентов (размер популяции пингвинов).
    max_iter : int
        Максимальное число итераций алгоритма.
    seed : int, optional
        Seed для инициализации генератора случайных чисел.
    processes : int, optional
        Количество процессов в пуле для параллельного вычисления.


    Attributes
    ----------
    penguins : np.ndarray, shape (agents, dim)
        Текущие позиции всех агентов (пингвинов).
    fitness : np.ndarray, shape (agents,)
        Значения целевой функции для каждой позиции.
    best_penguin : np.ndarray, shape (dim,)
        Лучшая найденная позиция (минимум среди агентов).
    best_fitness : float
        Значение целевой функции в best_penguin.
    iteration : int
        Текущий номер итерации.
    _processes : multiprocessing.Pool
        Пул рабочих процессов для параллельного расчёта фитнеса.

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
        # Сохраняем функцию оптимизации и размерность задачи
        self.fitness_function = self.problem
        self.problem_dimen = self.dim
        self._bounds = self.bounds
        self.n_penguins = self.agents
        self._max_iter = self.max_iter

        # Инициализируем генератор случайных чисел при наличии seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self._processes = self.params.get('processes', None)

        # Определяем начальные границы поиска
        self._bounds = initialize_bounds(self.problem_dimen, self._bounds)

        # Генерируем начальные позиции пингвинов равномерно в пределах границ
        self.penguins = initialize_positions(self.n_penguins, self.problem_dimen, self._bounds, self.seed)

        # Оцениваем начальный фитнес всех пингвинов параллельно
        self.fitness = mp_evaluate_fitness(self.penguins, self.fitness_function, self._processes)
        # Определяем лучшего пингвина (минимум функции
        self.best_penguin = self.penguins[np.argmin(self.fitness)]
        self.best_fitness = float(np.min(self.fitness))
        self.iteration = 0

    def iterate(self) -> None:
        """
        Одна итерация EPO обновляет позиции пчел.
        """
        # Обновляем параметр a, уменьшающийся от 2 до 0
        a = 2 - 2 * (self.iteration / self.max_iter)
        # Случайный радиус huddle и температура
        R = np.random.rand()  # Радиус скопления
        T = 1 if R <= 1 else 0  # Температура внутри скопления
        T_prime = T - (self.iteration / self.max_iter)  # Скорректированная температура

        for i in range(self.n_penguins):
            # Расстояние от пингвина до глобального лидера
            Pgrid = abs(self.best_penguin - self.penguins[i])

            # Вектор A (моделирует движение в сторону лидера с учётом температуры)
            A_vec = (2 * T_prime + Pgrid * np.random.rand(self.problem_dimen)) - T_prime

            # Случайный вектор C
            C_vec = np.random.rand(self.problem_dimen)

            # Вектор D (модулированное расстояние)
            D_vec = abs(C_vec * self.best_penguin - self.penguins[i])

            # Обновление позиции
            new_pos = self.penguins[i] + A_vec * D_vec

            # Проверяем границы поиска
            new_pos = enforce_boundaries(new_pos, self._bounds)

            # Распределённая оценка фитнеса после обновления
            new_fitness = self.fitness_function(new_pos)

            if new_fitness < self.fitness[i]:
                self.penguins[i] = new_pos
                self.fitness[i] = new_fitness

                if new_fitness < self.best_fitness:
                    self.best_fitness = float(new_fitness)
                    self.best_penguin = new_pos.copy()

    def get_result(self) -> Dict[str, Any]:
        """
        Возвращает словарь с лучшим найденным решением.
        """
        return {"minimum_x": self.best_penguin.tolist(), "minimum_value": float(self.best_fitness)}
