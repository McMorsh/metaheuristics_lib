"""Emperor Penguin Optimizer (EPO)."""

import numpy as np

from metaheuristics_lib.core.algorithm import BaseAlgorithm


class EmperorPenguinOptimizer(BaseAlgorithm):
    """Метаэвристический алгоритм Emperor Penguin Optimizer.

    Алгоритм моделирует поведение императорских пингвинов при
    совместном поиске пищи и сохранении тепла.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число пингвинов.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.

    Attributes:
        penguins: текущие позиции пингвинов формы ``(agents, dim)``.
        fitness: значения функции для текущих позиций.
    """

    def initialize(self) -> None:
        """Инициализирует популяцию пингвинов."""
        self._reset_state()
        self.penguins = self._initial_positions()
        self.fitness = self._evaluate(self.penguins)
        self._update_best(self.penguins, self.fitness)
        self.iteration = 0

    def iterate(self) -> None:
        """Выполняет одну итерацию EPO.

        Все новые позиции строятся от лучшего решения на начало
        итерации и оцениваются одним пакетом. Обновления внутри
        итерации не влияют на кандидатов других пингвинов.
        """
        progress = self.iteration / self.max_iter
        # В исходной формуле R равномерно в [0, 1], поэтому T всегда 1.
        radius = self.rng.random()
        temperature = 1.0 if radius <= 1.0 else 0.0
        adjusted_temperature = temperature - progress

        best = self._best_position()
        distance = np.abs(best - self.penguins)
        move = (
            2.0 * adjusted_temperature
            + distance * self.rng.random((self.agents, self.dim))
            - adjusted_temperature
        )
        random_vector = self.rng.random((self.agents, self.dim))
        modulated = np.abs(random_vector * best - self.penguins)
        candidates = self._clip(self.penguins + move * modulated)

        values = self._evaluate(candidates)
        improved = values < self.fitness
        self.penguins[improved] = candidates[improved]
        self.fitness[improved] = values[improved]
        self._update_best(self.penguins, self.fitness)
        self.iteration += 1
