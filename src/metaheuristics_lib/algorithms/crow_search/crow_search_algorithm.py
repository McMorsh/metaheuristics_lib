"""Crow Search Algorithm (CSA)."""

import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.algorithm import BaseAlgorithm


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
        flight_length: длина полёта, по умолчанию 1.0.
        awareness_prob: вероятность обнаружения спойлера.
        expand_rate: коэффициент расширения границ, по умолчанию 1.2.

    Attributes:
        crows: текущие позиции ворон формы ``(agents, dim)``.
        best_crows: лучшие запомненные позиции каждой вороны.
        fitness: лучшие значения функции каждой вороны.
    """

    ALLOWED_PARAMS = frozenset(
        {"awareness_prob", "expand_rate", "flight_length"},
    )

    def initialize(self) -> None:
        """Инициализирует популяцию и память ворон.

        Raises:
            ValueError: если вероятность осознания вне диапазона [0, 1].
        """
        self._reset_state()

        self.fly_len = float(self.params.get("flight_length", 1.0))
        self.awar_prob = float(self.params.get("awareness_prob", 0.1))
        self.expand_rate = float(self.params.get("expand_rate", 1.2))
        if not 0.0 <= self.awar_prob <= 1.0:
            raise ValueError(
                "Параметр awareness_prob должен быть в диапазоне [0, 1], "
                f"получено {self.awar_prob}"
            )
        if self.expand_rate < 1.0:
            raise ValueError(
                "Параметр expand_rate должен быть не меньше 1, "
                f"получено {self.expand_rate}"
            )

        self.crows = self._initial_positions()
        self.best_crows = self.crows.copy()
        self.fitness = self._evaluate(self.crows)
        self._update_best(self.best_crows, self.fitness)
        self.iteration = 0

    def _clip_with_expansion(
        self,
        position: npt.NDArray[np.float64],
        expand_rate: float,
    ) -> npt.NDArray[np.float64]:
        """Ограничивает позицию, расширяя границы как в исходной CSA."""
        updated = self.bounds.copy()
        for index in range(self.dim):
            low, high = updated[index]
            value = position[index]
            if value < low:
                updated[index, 0] = value * expand_rate
            elif value > high:
                updated[index, 1] = value * expand_rate
        self.bounds = updated
        return np.clip(position, updated[:, 0], updated[:, 1])

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
            new_crows[index] = self._clip_with_expansion(
                new_crows[index],
                self.expand_rate,
            )

        new_fitness = self._evaluate(new_crows)
        improved = new_fitness < self.fitness
        self.best_crows[improved] = new_crows[improved]
        self.fitness[improved] = new_fitness[improved]
        self.crows = new_crows.copy()
        self._update_best(self.best_crows, self.fitness)
        self.iteration += 1
