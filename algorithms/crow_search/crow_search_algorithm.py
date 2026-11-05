"""Crow Search Algorithm (CSA)."""

import warnings

import numpy as np

from core.algorithm import BaseAlgorithm


class CrowSearchAlgorithm(BaseAlgorithm):
    """Метаэвристический алгоритм Crow Search Algorithm.

    Алгоритм моделирует поведение ворон, которые прячут пищу и
    наблюдают за другими воронами.

    Args:
        problem: целевая функция.
        dim: размерность пространства поиска.
        bounds: границы поиска.
        agents: число ворон.
        max_iter: максимальное число итераций.
        seed: seed генератора случайных чисел.
        flight_length: длина полёта, по умолчанию 2.0.
        awareness_prob: вероятность обнаружения спойлера.

    Attributes:
        crows: текущие позиции ворон формы ``(agents, dim)``.
        best_crows: лучшие запомненные позиции каждой вороны.
        fitness: значения функции для текущих позиций.
        memory_fitness: лучшие значения функции каждой вороны.
    """

    def initialize(self) -> None:
        """Инициализирует популяцию и память ворон.

        Raises:
            ValueError: если вероятность осознания вне диапазона [0, 1].
        """
        self._reset_state()
        if "expand_rate" in self.params:
            warnings.warn(
                "Параметр expand_rate устарел: CSA использует "
                "фиксированные границы",
                DeprecationWarning,
                stacklevel=2,
            )

        self.fly_len = float(self.params.get("flight_length", 2.0))
        self.awar_prob = float(self.params.get("awareness_prob", 0.1))
        if not 0.0 <= self.awar_prob <= 1.0:
            raise ValueError(
                "Параметр awareness_prob должен быть в диапазоне [0, 1], "
                f"получено {self.awar_prob}"
            )

        self.crows = self._initial_positions()
        self.best_crows = self.crows.copy()
        self.fitness = self._evaluate(self.crows)
        self.memory_fitness = self.fitness.copy()
        self._update_best(self.best_crows, self.memory_fitness)
        self.iteration = 0

    def iterate(self) -> None:
        """Выполняет одну итерацию CSA."""
        new_crows = np.empty_like(self.crows)
        targets = self.rng.integers(
            low=0,
            high=self.agents,
            size=self.agents,
        )

        for index in range(self.agents):
            if self.rng.random() > self.awar_prob:
                target = self.best_crows[targets[index]]
                new_crows[index] = self.crows[index] + (
                    self.fly_len
                    * self.rng.random()
                    * (target - self.crows[index])
                )
            else:
                new_crows[index] = self.rng.uniform(
                    low=self.bounds[:, 0],
                    high=self.bounds[:, 1],
                )
            new_crows[index] = self._clip(new_crows[index])

        new_fitness = self._evaluate(new_crows)
        improved = new_fitness < self.fitness
        self.best_crows[improved] = new_crows[improved]
        self.memory_fitness[improved] = new_fitness[improved]
        self.fitness[improved] = new_fitness[improved]
        self.crows = new_crows.copy()
        self._update_best(self.best_crows, self.memory_fitness)
        self.iteration += 1
