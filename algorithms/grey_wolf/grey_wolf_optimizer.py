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
        self.fitness = self._evaluate(self.wolves)
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
        """Выполняет одну итерацию GWO.

        Расчёт лидеров и координат векторизован по агентам, размерностям
        и трём лидерам, чтобы убрать двойной цикл Python.
        """
        a = 2.0 - 2.0 * self.iteration / self.max_iter
        draws = self.rng.random((self.agents, self.dim, 3, 2))
        r1 = draws[:, :, :, 0]
        r2 = draws[:, :, :, 1]

        leaders = np.stack(
            [self.alpha_pos, self.beta_pos, self.delta_pos],
        )
        leaders = leaders.T[None, :, :]
        wolves = self.wolves[:, :, None]

        coefficient_a = a * (2.0 * r1 - 1.0)
        distance = np.abs(2.0 * r2 * leaders - wolves)
        candidates = leaders - coefficient_a * distance

        self.wolves = self._clip(np.mean(candidates, axis=2))
        self.fitness = self._evaluate(self.wolves)
        self._rank_leaders()
        self._update_best(self.wolves, self.fitness)
        self.iteration += 1
