"""Тесты проверки границ."""

import numpy as np
import pytest

from metaheuristics_lib.core.types import normalize_bounds


def test_common_interval_is_tiled() -> None:
    bounds = normalize_bounds((-1.0, 1.0), 3)
    np.testing.assert_allclose(
        bounds,
        np.tile([-1.0, 1.0], (3, 1)),
    )


def test_per_dimension_bounds_are_accepted() -> None:
    bounds = normalize_bounds([(-1.0, 1.0), (0.0, 10.0)], 2)
    np.testing.assert_allclose(
        bounds,
        [[-1.0, 1.0], [0.0, 10.0]],
    )


def test_ndarray_is_accepted() -> None:
    source = np.array([[-2.0, 2.0], [3.0, 4.0]])
    bounds = normalize_bounds(source, 2)
    np.testing.assert_allclose(bounds, source)


def test_result_is_a_copy() -> None:
    source = np.array([[-1.0, 1.0]])
    bounds = normalize_bounds(source, 1)
    bounds[0, 0] = -5.0
    assert source[0, 0] == -1.0


@pytest.mark.parametrize(
    ("bounds", "dim"),
    [
        ((0.0, 1.0), 0),
        ((1.0, 1.0), 2),
        ((0.0, 1.0, 2.0), 2),
        ([(0.0, 1.0)], 2),
        ([(0.0, 1.0), (2.0, 1.0)], 2),
        ((float("nan"), 1.0), 2),
        ((0.0, float("inf")), 2),
        ([[0.0, float("nan")], [0.0, 1.0]], 2),
    ],
)
def test_invalid_bounds_raise(bounds: object, dim: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_bounds(bounds, dim)  # type: ignore[arg-type]
