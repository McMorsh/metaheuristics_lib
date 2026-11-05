"""Тесты вспомогательных функций для популяций."""

import numpy as np
import pytest

from metaheuristics_lib.problems.continuous import sphere
from metaheuristics_lib.utils.algorithm_utils import (
    enforce_boundaries,
    enforce_boundaries_csa,
    evaluate_fitness_serial,
    initialize_bounds,
    initialize_positions,
)

BOUNDS = [(-1.0, 1.0), (0.0, 2.0)]


def test_initialize_bounds_returns_matrix() -> None:
    bounds = initialize_bounds(2, BOUNDS)
    assert bounds.shape == (2, 2)


def test_initialize_positions_are_reproducible() -> None:
    first = initialize_positions(4, 2, BOUNDS, seed=42)
    second = initialize_positions(4, 2, BOUNDS, seed=42)
    np.testing.assert_allclose(first, second)


def test_initialize_positions_accept_generator() -> None:
    first = initialize_positions(
        4,
        2,
        BOUNDS,
        seed=np.random.default_rng(42),
    )
    second = initialize_positions(4, 2, BOUNDS, seed=42)
    np.testing.assert_allclose(first, second)


def test_initialize_positions_respect_bounds() -> None:
    positions = initialize_positions(100, 2, BOUNDS, seed=42)
    assert np.all(positions[:, 0] >= -1.0)
    assert np.all(positions[:, 0] <= 1.0)
    assert np.all(positions[:, 1] >= 0.0)
    assert np.all(positions[:, 1] <= 2.0)


def test_initialize_positions_rejects_invalid_agents() -> None:
    with pytest.raises(ValueError, match="agents"):
        initialize_positions(0, 2, BOUNDS)


def test_initialize_positions_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        initialize_positions(4, 0, BOUNDS)


def test_enforce_boundaries_clips_positions() -> None:
    positions = np.array([[-2.0, 3.0], [0.0, 1.0]])
    clipped = enforce_boundaries(positions, BOUNDS)
    np.testing.assert_allclose(
        clipped,
        [[-1.0, 2.0], [0.0, 1.0]],
    )


def test_enforce_boundaries_csa_expands_upper_bound() -> None:
    clipped, updated = enforce_boundaries_csa(
        np.array([2.0]),
        np.array([[-1.0, 1.0]]),
    )
    assert clipped[0] == pytest.approx(2.0)
    np.testing.assert_allclose(updated, [[-1.0, 2.2]])


def test_enforce_boundaries_csa_handles_negative_bounds() -> None:
    clipped, updated = enforce_boundaries_csa(
        np.array([-3.9]),
        np.array([[-5.0, -4.0]]),
    )
    assert clipped[0] == pytest.approx(-3.9)
    np.testing.assert_allclose(updated, [[-5.0, -3.8]])


def test_enforce_boundaries_csa_expands_lower_bound() -> None:
    clipped, updated = enforce_boundaries_csa(
        np.array([4.9]),
        np.array([[5.0, 6.0]]),
    )
    assert clipped[0] == pytest.approx(4.9)
    np.testing.assert_allclose(updated, [[4.8, 6.0]])


def test_enforce_boundaries_csa_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="expand_rate"):
        enforce_boundaries_csa(
            np.array([0.0]),
            np.array([[-1.0, 1.0]]),
            expand_rate=0.9,
        )


def test_enforce_boundaries_csa_rejects_matrix() -> None:
    with pytest.raises(ValueError, match="одномерным"):
        enforce_boundaries_csa(
            np.zeros((2, 2)),
            BOUNDS,
        )


def test_evaluate_fitness_serial_matches_direct_calls() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 2.0]])

    values = evaluate_fitness_serial(positions, sphere)

    np.testing.assert_allclose(values, [0.0, 5.0])
