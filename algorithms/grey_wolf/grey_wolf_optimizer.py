"""Grey Wolf Optimizer (GWO)."""

import numpy as np

from core.algorithm import BaseAlgorithm


class GreyWolfOptimizer(BaseAlgorithm):
    """Метаэвристический алгоритм Grey Wolf Optimizer.

    Алгоритм моделирует иерархию и охоту серых волков. Популяция
    обновляется относительно лидеров alpha, beta и delta.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число волков, не меньше 3.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.

    Attributes:
        wolves: позиции волков формы ``(agents, dim)``.
        fitness: значения функции для текущих позиций.
        alpha_pos: позиция лучшего волка.
        beta_pos: позиция второго волка.
        delta_pos: позиция третьего волка.
        alpha_fit: значение функции для alpha.
        beta_fit: значение функции для beta.
        delta_fit: значение функции для delta.
    """

    def initialize(self) -> None:
        """Инициализирует популяцию и лидеров.

        Raises:
            ValueError: если волков меньше трёх.
        """
        self._reset_state()
        if self.agents < 3:
            raise ValueError(
                f"Для GWO требуется хотя бы три волка, получено {self.agents}"
            )

        self.wolves = self._initial_positions()
        self.fitness = self.evaluator.evaluate(self.wolves)
        self._rank_leaders()
        self._update_best(self.wolves, self.fitness)
        self.iteration = 0

    def _rank_leaders(self) -> None:
        """Выбирает alpha, beta и delta по текущему fitness."""
        order = np.argsort(self.fitness)
        self.alpha_pos = self.wolves[order[0]].copy()
        self.alpha_fit = float(self.fitness[order[0]])
        self.beta_pos = self.wolves[order[1]].copy()
        self.beta_fit = float(self.fitness[order[1]])
        self.delta_pos = self.wolves[order[2]].copy()
        self.delta_fit = float(self.fitness[order[2]])

    def iterate(self) -> None:
        """Выполняет одну итерацию GWO."""
        a = 2.0 - 2.0 * self.iteration / self.max_iter

        for index in range(self.agents):
            for dimension in range(self.dim):
                r1 = self.rng.random()
                r2 = self.rng.random()
                a1 = 2.0 * a * r1 - a
                c1 = 2.0 * r2
                distance_alpha = abs(
                    c1 * self.alpha_pos[dimension]
                    - self.wolves[index, dimension]
                )
                x1 = self.alpha_pos[dimension] - a1 * distance_alpha

                r1 = self.rng.random()
                r2 = self.rng.random()
                a2 = 2.0 * a * r1 - a
                c2 = 2.0 * r2
                distance_beta = abs(
                    c2 * self.beta_pos[dimension]
                    - self.wolves[index, dimension]
                )
                x2 = self.beta_pos[dimension] - a2 * distance_beta

                r1 = self.rng.random()
                r2 = self.rng.random()
                a3 = 2.0 * a * r1 - a
                c3 = 2.0 * r2
                distance_delta = abs(
                    c3 * self.delta_pos[dimension]
                    - self.wolves[index, dimension]
                )
                x3 = self.delta_pos[dimension] - a3 * distance_delta

                self.wolves[index, dimension] = (x1 + x2 + x3) / 3.0

            self.wolves[index] = self._clip(self.wolves[index])

        self.fitness = self.evaluator.evaluate(self.wolves)
        self._rank_leaders()
        self._update_best(self.wolves, self.fitness)
        self.iteration += 1
