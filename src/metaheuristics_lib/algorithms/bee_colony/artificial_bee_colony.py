"""Artificial Bee Colony (ABC)."""

import numpy as np

from metaheuristics_lib.core.algorithm import BaseAlgorithm


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

    ALLOWED_PARAMS = frozenset({"limit"})

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
        self.fitness = self._evaluate(self.bees)
        self.local_best_bee = self.bees.copy()
        self.local_best_fitness = self.fitness.copy()
        self.trial = np.zeros(self.agents, dtype=np.int64)
        self._update_best(self.bees, self.fitness)
        self.iteration = 0

    def _neighbour(self, index: int) -> None:
        """Генерирует и оценивает соседнее решение для одной пчелы."""
        partner = int(self.rng.integers(0, self.agents - 1))
        if partner >= index:
            partner += 1
        phi = self.rng.uniform(-1.0, 1.0, size=self.dim)
        candidate = self.bees[index] + phi * (
            self.bees[index] - self.bees[partner]
        )
        candidate = self._clip(candidate)
        value = float(self._evaluate(candidate.reshape(1, self.dim))[0])
        if value < self.local_best_fitness[index]:
            self.bees[index] = candidate
            self.fitness[index] = value
            self.local_best_bee[index] = candidate
            self.local_best_fitness[index] = value
            self.trial[index] = 0
            self._consider_best(candidate, value)
        else:
            self.trial[index] += 1

    def iterate(self) -> None:
        """Выполняет одну итерацию ABC.

        Фазы выполняются последовательно: каждая следующая пчела видит
        обновления, сделанные предыдущими. Наблюдатели выбираются
        пропорционально ``1 / (1 + abs(fitness))``, как в исходной
        версии алгоритма.
        """
        for index in range(self.agents):
            self._neighbour(index)

        probabilities = 1.0 / (1.0 + np.abs(self.local_best_fitness))
        probabilities /= np.sum(probabilities)
        for _ in range(self.agents):
            index = int(self.rng.choice(self.agents, p=probabilities))
            self._neighbour(index)

        scout_indices = np.flatnonzero(self.trial > self._limit)
        for scout_index in scout_indices:
            position = self.rng.uniform(
                low=self.bounds[:, 0],
                high=self.bounds[:, 1],
            )
            value = float(self._evaluate(position.reshape(1, self.dim))[0])
            self.bees[scout_index] = position
            self.fitness[scout_index] = value
            self.local_best_bee[scout_index] = position
            self.local_best_fitness[scout_index] = value
            self.trial[scout_index] = 0
            self._consider_best(position, value)

        self._update_best(
            self.local_best_bee,
            self.local_best_fitness,
        )
        self.iteration += 1
