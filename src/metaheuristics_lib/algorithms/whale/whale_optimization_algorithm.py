"""Whale Optimization Algorithm (WOA)."""

import numpy as np

from metaheuristics_lib.core.algorithm import BaseAlgorithm


class WhaleOptimizationAlgorithm(BaseAlgorithm):
    """Метаэвристический алгоритм Whale Optimization Algorithm.

    Алгоритм моделирует охоту горбатых китов: обволакивание жертвы
    и спиральное движение.

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
        self.fitness = self._evaluate(self.whales)
        self._update_best(self.whales, self.fitness)
        self.iteration = 0

    def iterate(self) -> None:
        """Выполняет одну итерацию WOA.

        Используется исходная логика без отдельной исследовательской
        ветви: при ``p < 0.5`` кит движется к лучшему решению, иначе
        выполняет спиральное движение.
        """
        a = 2.0 - 2.0 * self.iteration / self.max_iter
        best = self._best_position()

        for index in range(self.agents):
            coefficient_a = 2.0 * a * self.rng.random() - a
            coefficient_c = 2.0 * self.rng.random()
            probability = self.rng.random()

            if probability < 0.5:
                distance = np.abs(coefficient_c * best - self.whales[index])
                new_position = best - coefficient_a * distance
            else:
                distance = np.abs(best - self.whales[index])
                spiral = self.rng.uniform(-1.0, 1.0)
                new_position = (
                    distance * np.exp(spiral) * np.cos(2.0 * np.pi * spiral)
                    + best
                )

            self.whales[index] = self._clip(new_position)

        self.fitness = self._evaluate(self.whales)
        self._update_best(self.whales, self.fitness)
        self.iteration += 1
