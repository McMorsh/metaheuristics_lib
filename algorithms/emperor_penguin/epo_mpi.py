from typing import Dict, Any

import numpy as np

from core.algorithm import BaseAlgorithm
from utils.algorithm_utils import initialize_bounds
from utils.mpi_utils import mpi_evaluate_fitness, rank


class EmperorPenguinOptimizerMPI(BaseAlgorithm):
    """
    Emperor Penguin Optimizer with mpi4py

    Реализует метаэвристический алгоритм глобальной оптимизации,
    вдохновлённый социальным поведением императорских пингвинов
    при совместном поиске пищи и сохранении тепла.

    Распараллеливание через MPI позволяет распределять вычисление фитнеса
    и синхронизацию позиций между процессами.

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
        if rank == 0:
            # Сохраняем функцию оптимизации и размерность задачи
            self.fitness_function = self.problem
            self.problem_dimen = self.dim
            self._bounds = np.array(self.bounds)
            self.n_penguins = self.agents
            self._max_iter = self.max_iter

            # Инициализируем генератор случайных чисел при наличии seed
            if self.seed is not None:
                np.random.seed(self.seed)

            # Определяем начальные границы поиска
            self.low_bounds, self.high_bounds = initialize_bounds(self.problem_dimen, self._bounds[0][0],
                                                                  self._bounds[0][1])

            # Генерируем начальные позиции пингвинов равномерно в пределах границ
            self.penguins = np.random.uniform(low=self.low_bounds,
                                              high=self.high_bounds,
                                              size=(self.n_penguins, self.problem_dimen))

        # Оцениваем начальный фитнес всех пингвинов параллельно
        self.fitness = mpi_evaluate_fitness(self.penguins, self.fitness_function)

        if rank == 0:
            # Определяем лучшего пингвина (минимум функции)
            self.best_penguin = self.penguins[np.argmin(self.fitness)]
            self.best_fitness = float(np.min(self.fitness))
            self.iteration = 0

    def iterate(self) -> None:
        """
        Одна итерация EPO обновляет позиции пчел.
        """
        if rank == 0:
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
            new_pos = np.clip(new_pos, self.low_bounds, self.high_bounds)

            # Распределённая оценка фитнеса после обновления
            score = mpi_evaluate_fitness(new_pos, self.fitness_function)

            self.penguins[i] = new_pos
            self.fitness[i] = score

            if score < self.best_score:
                self.best_score = score
                self.best_penguin = new_pos.copy()

    def get_result(self) -> Dict[str, Any]:
        """
        Возвращает словарь с лучшим найденным решением.
        """
        if rank == 0:
            return {"minimum_x": self.best_penguin.tolist(), "minimum_value": float(self.best_fitness)}
        return None
