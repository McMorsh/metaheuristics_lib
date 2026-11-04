"""Artificial Bee Colony (ABC)."""

import numpy as np

from core.algorithm import BaseAlgorithm


class ArtificialBeeColony(BaseAlgorithm):
    """Метаэвристический алгоритм Artificial Bee Colony.

    Алгоритм моделирует поведение рабочих, наблюдающих и скаут-пчёл.
    Минимизирует целевую функцию, переданную в ``problem``.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число пчёл, не меньше 2.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        limit: число итераций без улучшения до превращения в скаута.

    Attributes:
        bees: текущие позиции пчёл формы ``(agents, dim)``.
        fitness: текущие значения функции.
        local_best_bee: лучшие позиции каждой пчелы.
        local_best_fitness: лучшие значения функции каждой пчелы.
        trial: счётчики безуспешных попыток.
    """

    def initialize(self) -> None:
        """Инициализирует популяцию и память пчёл.

        Raises:
            ValueError: если пчёл меньше двух или ``limit < 1``.
        """
        self._reset_state()
        if self.agents < 2:
            raise ValueError(
                f"Для ABC требуется хотя бы две пчелы, получено {self.agents}"
            )

        limit = int(self.params.get("limit", 25))
        if limit < 1:
            raise ValueError(
                f"Параметр limit должен быть положительным, получено {limit}"
            )
        self._limit = limit

        self.bees = self._initial_positions()
        self.fitness = self.evaluator.evaluate(self.bees)
        self.local_best_bee = self.bees.copy()
        self.local_best_fitness = self.fitness.copy()
        self.trial = np.zeros(self.agents, dtype=np.int64)
        self._update_best(self.bees, self.fitness)
        self.iteration = 0

    def _neighbour(self, index: int) -> None:
        """Обновляет источник пищи по правилам рабочей пчелы."""
        other = int(self.rng.integers(0, self.agents - 1))
        if other >= index:
            other += 1

        phi = self.rng.uniform(-1.0, 1.0, self.dim)
        candidate = self.bees[index] + phi * (
            self.bees[index] - self.bees[other]
        )
        candidate = self._clip(candidate)
        value = float(
            self.evaluator.evaluate(candidate.reshape(1, self.dim))[0]
        )

        if value < self.local_best_fitness[index]:
            self.bees[index] = candidate
            self.fitness[index] = value
            self.local_best_bee[index] = candidate
            self.local_best_fitness[index] = value
            self.trial[index] = 0
        else:
            self.trial[index] += 1

    def iterate(self) -> None:
        """Выполняет одну итерацию ABC."""
        for index in range(self.agents):
            self._neighbour(index)

        shifted = self.local_best_fitness - np.min(self.local_best_fitness)
        weights = 1.0 / (shifted + np.finfo(np.float64).eps)
        probabilities = weights / np.sum(weights)

        for _ in range(self.agents):
            index = int(self.rng.choice(self.agents, p=probabilities))
            self._neighbour(index)

        scout_indices = np.flatnonzero(self.trial > self._limit)
        if scout_indices.size:
            low = self.bounds[:, 0]
            high = self.bounds[:, 1]
            positions = self.rng.uniform(
                low=low,
                high=high,
                size=(scout_indices.size, self.dim),
            )
            values = self.evaluator.evaluate(positions)
            for position_index, scout_index in enumerate(scout_indices):
                position = positions[position_index]
                value = float(values[position_index])
                self.bees[scout_index] = position
                self.fitness[scout_index] = value
                self.local_best_bee[scout_index] = position.copy()
                self.local_best_fitness[scout_index] = value
                self.trial[scout_index] = 0

        self._update_best(
            self.local_best_bee,
            self.local_best_fitness,
        )
        self.iteration += 1
