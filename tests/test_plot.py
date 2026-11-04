"""Тесты построения графиков."""

from pathlib import Path

import pytest

from problems.continuous import sphere
from utils.plot import (
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


def test_plot_convergence_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="пустым"):
        plot_convergence([])


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


def test_plot_speedup_saves_file(tmp_path: Path) -> None:
    path = tmp_path / "speedup.png"
    plot_speedup_different_pools(
        [2.0, 1.0],
        [1, 2],
        save_path=path,
    )
    assert path.exists()


def test_plot_speedup_rejects_zero_base() -> None:
    with pytest.raises(ValueError, match="Базовое время"):
        plot_speedup_different_pools([0.0, 1.0], [1, 2])


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
