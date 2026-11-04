"""Whale Optimization Algorithm (WOA)."""

import numpy as np

from core.algorithm import BaseAlgorithm


class WhaleOptimizationAlgorithm(BaseAlgorithm):
    """Метаэвристический алгоритм Whale Optimization Algorithm.

    Алгоритм моделирует охоту горбатых китов: обволакивание жертвы,
    спиральное движение и исследование пространства.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число китов.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.

    Attributes:
        whales: текущие позиции китов формы ``(agents, dim)``.
        fitness: значения функции для текущих позиций.
    """

    def initialize(self) -> None:
        """Инициализирует популяцию китов."""
        self._reset_state()
        self.whales = self._initial_positions()
        self.fitness = self.evaluator.evaluate(self.whales)
        self._update_best(self.whales, self.fitness)
        self.iteration = 0

    def iterate(self) -> None:
        """Выполняет одну итерацию WOA."""
        a = 2.0 - 2.0 * self.iteration / self.max_iter
        best = self._best_position()

        for index in range(self.agents):
            coefficient_a = 2.0 * a * self.rng.random() - a
            coefficient_c = 2.0 * self.rng.random()
            probability = self.rng.random()

            if probability < 0.5:
                if abs(coefficient_a) < 1.0:
                    distance = np.abs(
                        coefficient_c * best - self.whales[index]
                    )
                    new_position = best - coefficient_a * distance
                else:
                    random_index = int(self.rng.integers(0, self.agents))
                    random_whale = self.whales[random_index]
                    distance = np.abs(
                        coefficient_c * random_whale - self.whales[index]
                    )
                    new_position = random_whale - coefficient_a * distance
            else:
                distance = np.abs(best - self.whales[index])
                spiral = self.rng.uniform(-1.0, 1.0)
                new_position = (
                    distance * np.exp(spiral) * np.cos(2.0 * np.pi * spiral)
                    + best
                )

            self.whales[index] = self._clip(new_position)

        self.fitness = self.evaluator.evaluate(self.whales)
        self._update_best(self.whales, self.fitness)
        self.iteration += 1
