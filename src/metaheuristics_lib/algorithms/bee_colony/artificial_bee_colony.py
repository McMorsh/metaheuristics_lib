"""Artificial Bee Colony (ABC)."""

import numpy as np
import numpy.typing as npt

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

    def _select_partners(
        self,
        indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Возвращает партнёра для каждой выбранной пчелы."""
        partners = np.asarray(
            self.rng.integers(0, self.agents - 1, size=indices.size),
            dtype=np.int64,
        )
        return partners + (partners >= indices)

    def _propose_candidates(
        self,
        indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """Строит кандидатов по состоянию популяции на начало фазы."""
        partners = self._select_partners(indices)
        phi = self.rng.uniform(-1.0, 1.0, size=(indices.size, self.dim))
        base = self.bees[indices]
        return self._clip(base + phi * (base - self.bees[partners]))

    def _apply_candidates(
        self,
        indices: npt.NDArray[np.int64],
        candidates: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        """Применяет улучшения и обновляет счётчики попыток."""
        for position, index in enumerate(indices):
            value = float(values[position])
            if value < self.local_best_fitness[index]:
                self.bees[index] = candidates[position]
                self.fitness[index] = value
                self.local_best_bee[index] = candidates[position]
                self.local_best_fitness[index] = value
                self.trial[index] = 0
            else:
                self.trial[index] += 1

    def iterate(self) -> None:
        """Выполняет одну итерацию ABC.

        В отличие от последовательного варианта, рабочие и наблюдающие
        пчёлы оценивают пакет кандидатов, построенных по состоянию на
        начало фазы. Кандидат не видит обновлений, сделанных другими
        пчёлами в той же фазе.
        """
        indices = np.arange(self.agents, dtype=np.int64)
        candidates = self._propose_candidates(indices)
        values = self._evaluate(candidates)
        self._apply_candidates(indices, candidates, values)

        shifted = self.local_best_fitness - np.min(self.local_best_fitness)
        weights = 1.0 / (shifted + np.finfo(np.float64).eps)
        probabilities = weights / np.sum(weights)

        chosen = np.asarray(
            self.rng.choice(
                self.agents,
                size=self.agents,
                p=probabilities,
            ),
            dtype=np.int64,
        )
        candidates = self._propose_candidates(chosen)
        values = self._evaluate(candidates)
        self._apply_candidates(chosen, candidates, values)

        scout_indices = np.flatnonzero(self.trial > self._limit)
        if scout_indices.size:
            low = self.bounds[:, 0]
            high = self.bounds[:, 1]
            positions = self.rng.uniform(
                low=low,
                high=high,
                size=(scout_indices.size, self.dim),
            )
            values = self._evaluate(positions)
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
