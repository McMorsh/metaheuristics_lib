"""Emperor Penguin Optimizer (EPO)."""

import numpy as np

from core.algorithm import BaseAlgorithm


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
        self.fitness = self.evaluator.evaluate(self.penguins)
        self._update_best(self.penguins, self.fitness)
        self.iteration = 0

    def iterate(self) -> None:
        """Выполняет одну итерацию EPO."""
        progress = self.iteration / self.max_iter
        # В исходной формуле R равномерно в [0, 1], поэтому T всегда 1.
        radius = self.rng.random()
        temperature = 1.0 if radius <= 1.0 else 0.0
        adjusted_temperature = temperature - progress

        for index in range(self.agents):
            best = self._best_position()
            distance = np.abs(best - self.penguins[index])
            move = (
                2.0 * adjusted_temperature
                + distance * self.rng.random(self.dim)
                - adjusted_temperature
            )
            random_vector = self.rng.random(self.dim)
            modulated = np.abs(random_vector * best - self.penguins[index])
            new_position = self.penguins[index] + move * modulated
            new_position = self._clip(new_position)

            new_fitness = float(
                self.evaluator.evaluate(new_position.reshape(1, self.dim))[0]
            )
            if new_fitness < self.fitness[index]:
                self.penguins[index] = new_position
                self.fitness[index] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_position.copy()

        self.iteration += 1
