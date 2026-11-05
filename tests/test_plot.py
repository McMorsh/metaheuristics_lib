"""Тесты построения графиков."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from metaheuristics_lib.problems.continuous import sphere
from metaheuristics_lib.utils.plot import (
    plot_builder,
    plot_convergence,
    plot_execution_time_comparison,
    plot_fitness_function,
    plot_fitness_function_above,
    plot_speedup_different_pools,
)


def test_plot_convergence_saves_file(tmp_path: Path) -> None:
    path = tmp_path / "convergence.png"
    plot_convergence(
        [1.0, 0.5, 0.25],
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_accepts_many_histories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "convergence_many.png"
    plot_convergence(
        [[1.0, 0.5], [2.0, 1.5]],
        labels=["first", "second"],
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_accepts_numpy_history(tmp_path: Path) -> None:
    path = tmp_path / "convergence_numpy.png"
    plot_convergence(
        np.array([1.0, 0.5, 0.25]),
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_accepts_numpy_matrix(tmp_path: Path) -> None:
    path = tmp_path / "convergence_matrix.png"
    plot_convergence(
        np.array([[1.0, 0.5], [2.0, 1.5]]),
        labels=["first", "second"],
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_accepts_numpy_integer_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "convergence_numpy_int.png"
    plot_convergence(
        cast(Sequence[float], [np.int64(1), np.int64(2)]),
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="пустым"):
        plot_convergence([])


def test_plot_convergence_rejects_three_dimensional_array() -> None:
    with pytest.raises(ValueError, match="двумерными"):
        plot_convergence(np.zeros((2, 2, 2)))


def test_plot_convergence_fills_missing_labels(tmp_path: Path) -> None:
    path = tmp_path / "convergence_labels.png"
    plot_convergence(
        [[1.0, 0.5], [2.0, 1.0]],
        labels=["first"],
        save_path=path,
    )
    assert path.exists()


def test_plot_convergence_shows_without_save_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    try:
        plot_convergence([1.0, 0.5])
    finally:
        plt.close("all")

    assert shown == [True]


def test_plot_fitness_function_above_saves_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surface_top.png"
    plot_fitness_function_above(
        sphere,
        (-1.0, 1.0),
        resolution=5,
        save_path=path,
    )
    assert path.exists()


def test_plot_fitness_function_above_draws_best_point(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surface_top_best.png"
    plot_fitness_function_above(
        sphere,
        (-1.0, 1.0),
        resolution=5,
        best_pos=[0.0, 0.0],
        save_path=path,
    )
    assert path.exists()


def test_plot_fitness_function_draws_best_point(tmp_path: Path) -> None:
    path = tmp_path / "surface_3d_best.png"
    plot_fitness_function(
        sphere,
        (-1.0, 1.0),
        resolution=5,
        best_pos=[0.0, 0.0],
        save_path=path,
    )
    assert path.exists()


def test_plot_fitness_function_saves_file(tmp_path: Path) -> None:
    path = tmp_path / "surface_3d.png"
    plot_fitness_function(
        sphere,
        [(-1.0, 1.0), (-1.0, 1.0)],
        resolution=5,
        save_path=path,
    )
    assert path.exists()


def test_plot_execution_time_comparison_saves_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "times.png"
    plot_execution_time_comparison(
        [1.0, 2.0],
        ["first", "second"],
        save_path=path,
    )
    assert path.exists()


def test_plot_execution_time_comparison_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="совпадать"):
        plot_execution_time_comparison([1.0], ["first", "second"])


@pytest.mark.parametrize(
    "times",
    [
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
)
def test_plot_execution_time_comparison_rejects_nonpositive(
    times: list[float],
) -> None:
    with pytest.raises(ValueError, match="положительными"):
        plot_execution_time_comparison(
            times,
            ["first", "second"],
        )


def test_plot_execution_time_comparison_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="числами"):
        plot_execution_time_comparison(
            cast(Sequence[float], [1.0, "bad"]),
            ["first", "second"],
        )


def test_plot_execution_time_comparison_accepts_numpy_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "times_numpy.png"
    plot_execution_time_comparison(
        np.array([1.0, 2.0]),
        ["first", "second"],
        save_path=path,
    )
    assert path.exists()


def test_plot_speedup_saves_file(tmp_path: Path) -> None:
    path = tmp_path / "speedup.png"
    plot_speedup_different_pools(
        [2.0, 1.0],
        [1, 2],
        save_path=path,
    )
    assert path.exists()


def test_plot_speedup_accepts_numpy_times(tmp_path: Path) -> None:
    path = tmp_path / "speedup_numpy.png"
    plot_speedup_different_pools(
        np.array([2.0, 1.0]),
        [1, 2],
        save_path=path,
    )
    assert path.exists()


@pytest.mark.parametrize(
    "times",
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
)
def test_plot_speedup_rejects_nonpositive_time(
    times: list[float],
) -> None:
    with pytest.raises(ValueError, match="положительными"):
        plot_speedup_different_pools(times, [1, 2])


def test_plot_speedup_rejects_non_numeric_time() -> None:
    with pytest.raises(ValueError, match="числами"):
        plot_speedup_different_pools(
            cast(Sequence[float], [1.0, "bad"]),
            [1, 2],
        )


def test_plot_builder_returns_grid() -> None:
    grid_x, grid_y, grid_z, title = plot_builder(
        sphere,
        (-1.0, 1.0),
        resolution=5,
        title=None,
    )
    assert grid_x.shape == (5, 5)
    assert grid_y.shape == (5, 5)
    assert grid_z.shape == (5, 5)
    assert "sphere" in title


def test_plot_builder_rejects_zero_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        plot_builder(sphere, (-1.0, 1.0), resolution=0, title=None)
