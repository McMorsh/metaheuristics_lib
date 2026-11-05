"""Визуализация целевых функций и сходимости алгоритмов."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.types import (
    BoundsLike,
    Objective,
    normalize_bounds,
)

type SavePath = str | Path | None


def _save_or_show(save_path: SavePath) -> None:
    """Сохраняет текущий график или показывает его."""
    if save_path is None:
        plt.show()
        return

    path = Path(save_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _normalize_histories(
    histories: Sequence[float]
    | Sequence[Sequence[float]]
    | npt.NDArray[np.float64],
) -> list[list[float]]:
    """Приводит одну или несколько историй к списку списков."""
    if isinstance(histories, np.ndarray):
        array = np.asarray(histories, dtype=np.float64)
        if array.size == 0:
            raise ValueError("Список историй не должен быть пустым")
        if array.ndim == 1:
            return [[float(value) for value in array]]
        if array.ndim == 2:
            return [[float(value) for value in row] for row in array]
        raise ValueError("Истории должны быть одномерными или двумерными")

    items = list(histories)
    if not items:
        raise ValueError("Список историй не должен быть пустым")

    first = items[0]
    if isinstance(first, (int, float, np.integer, np.floating)):
        scalar_items = cast(Sequence[float], items)
        return [[float(value) for value in scalar_items]]

    sequence_items = cast(Sequence[Sequence[float]], items)
    return [[float(value) for value in history] for history in sequence_items]


def plot_builder(
    fitness_func: Objective,
    bounds: BoundsLike,
    resolution: int,
    title: str | None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    str,
]:
    """Строит сетку значений функции для двумерного графика.

    Args:
        fitness_func: целевая функция.
        bounds: границы поиска.
        resolution: число точек по каждой оси.
        title: заголовок или ``None`` для автоматического.

    Returns:
        Сетки ``X``, ``Y``, значения ``Z`` и заголовок.

    Raises:
        ValueError: если разрешение неположительно.
    """
    if resolution < 1:
        raise ValueError(
            "Параметр resolution должен быть положительным, "
            f"получено {resolution}"
        )

    bounds_array = normalize_bounds(bounds, 2)
    x_min, x_max = bounds_array[0]
    y_min, y_max = bounds_array[1]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    values = np.asarray(
        [fitness_func(point) for point in points],
        dtype=np.float64,
    )
    grid_z = values.reshape(grid_x.shape)

    function_name = getattr(
        fitness_func,
        "__name__",
        type(fitness_func).__name__,
    )
    resolved_title = title if title is not None else f"Функция {function_name}"
    return grid_x, grid_y, grid_z, resolved_title


def plot_fitness_function_above(
    fitness_func: Objective,
    bounds: BoundsLike,
    resolution: int = 100,
    title: str | None = None,
    best_pos: Sequence[float] | None = None,
    save_path: SavePath = None,
) -> None:
    """Строит контурный график функции.

    Args:
        fitness_func: целевая функция.
        bounds: границы поиска.
        resolution: число точек по каждой оси.
        title: заголовок графика.
        best_pos: координаты лучшего решения.
        save_path: путь сохранения или ``None`` для показа.
    """
    grid_x, grid_y, grid_z, resolved_title = plot_builder(
        fitness_func,
        bounds,
        resolution,
        title,
    )

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(contour)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(resolved_title)

    if best_pos is not None:
        point = np.atleast_1d(np.asarray(best_pos, dtype=np.float64))
        if point.size == 2:
            plt.plot(point[0], point[1], "r.", markersize=10)

    _save_or_show(save_path)


def plot_fitness_function(
    fitness_func: Objective,
    bounds: BoundsLike,
    resolution: int = 100,
    title: str | None = None,
    best_pos: Sequence[float] | None = None,
    save_path: SavePath = None,
) -> None:
    """Строит трёхмерный график функции.

    Args:
        fitness_func: целевая функция.
        bounds: границы поиска.
        resolution: число точек по каждой оси.
        title: заголовок графика.
        best_pos: координаты лучшего решения.
        save_path: путь сохранения или ``None`` для показа.
    """
    grid_x, grid_y, grid_z, resolved_title = plot_builder(
        fitness_func,
        bounds,
        resolution,
        title,
    )

    figure = plt.figure(figsize=(10, 7))
    axes = figure.add_subplot(111, projection="3d")
    surface = axes.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap="viridis",
        edgecolor="none",
    )
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Fitness Value")
    axes.set_title(resolved_title)
    figure.colorbar(surface, shrink=0.5, aspect=5)

    if best_pos is not None:
        point = np.atleast_1d(np.asarray(best_pos, dtype=np.float64))
        if point.size == 2:
            z_value = fitness_func(point)
            axes.scatter(
                point[0],
                point[1],
                z_value,
                color="r",
                s=100,
                label="Best Solution",
                marker="x",
            )
            axes.legend()

    _save_or_show(save_path)


def plot_convergence(
    histories: Sequence[float]
    | Sequence[Sequence[float]]
    | npt.NDArray[np.float64],
    labels: str | Sequence[str] | None = None,
    title: str = "График сравнения сходимости",
    save_path: SavePath = None,
) -> None:
    """Строит графики сходимости одного или нескольких алгоритмов.

    Args:
        histories: одна история или последовательность историй.
        labels: подпись или подписи алгоритмов.
        title: заголовок графика.
        save_path: путь сохранения или ``None`` для показа.

    Raises:
        ValueError: если список историй пуст или форма неверна.
    """
    history_list = _normalize_histories(histories)

    if isinstance(labels, str):
        label_list = [labels]
    elif labels is None:
        label_list = ["Лучшее значение"] * len(history_list)
    else:
        label_list = list(labels)
    while len(label_list) < len(history_list):
        label_list.append(f"Алгоритм {len(label_list) + 1}")

    plt.figure(figsize=(10, 6))
    for history, label in zip(
        history_list,
        label_list,
        strict=False,
    ):
        plt.plot(history, label=label)

    plt.title(title)
    plt.xlabel("Итерация")
    plt.ylabel("Значение целевой функции")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_execution_time_comparison(
    times: Sequence[float] | npt.NDArray[np.float64],
    labels: Sequence[str],
    title: str = "Сравнение времени выполнения алгоритмов",
    save_path: SavePath = None,
) -> None:
    """Строит столбчатую диаграмму времени выполнения.

    Args:
        times: время выполнения каждого алгоритма.
        labels: названия алгоритмов.
        title: заголовок графика.
        save_path: путь сохранения или ``None`` для показа.

    Raises:
        ValueError: если длины списков не совпадают, списки пусты,
            времена нечисловые или неположительные.
    """
    try:
        time_values = [float(value) for value in times]
    except (TypeError, ValueError) as error:
        raise ValueError("Времена должны быть числами") from error
    label_values = [str(label) for label in labels]
    if len(time_values) != len(label_values):
        raise ValueError(
            "Количество времён и названий алгоритмов должно совпадать"
        )
    if not time_values:
        raise ValueError("Список времён не должен быть пустым")
    if any(not np.isfinite(value) or value <= 0.0 for value in time_values):
        raise ValueError("Времена должны быть положительными")

    figure = plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_values)))
    bars = plt.bar(label_values, time_values, color=colors)
    plt.ylabel("Время (сек)", fontsize=18)
    plt.title(title, fontsize=18, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    max_time = max(time_values)
    plt.ylim(top=max_time * 1.20)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_time * 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=18,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.15)
    _save_or_show(save_path)


def plot_speedup_different_pools(
    times: Sequence[float] | npt.NDArray[np.float64],
    threads: Sequence[int],
    title: str = "График ускорения",
    save_path: SavePath = None,
) -> None:
    """Строит график ускорения относительно первого запуска.

    Args:
        times: время выполнения для каждого числа процессов.
        threads: число процессов.
        title: заголовок графика.
        save_path: путь сохранения или ``None`` для показа.

    Raises:
        ValueError: если длины списков не совпадают, списки пусты,
            времена нечисловые или хотя бы одно время неположительно.
    """
    try:
        time_values = [float(value) for value in times]
    except (TypeError, ValueError) as error:
        raise ValueError("Времена должны быть числами") from error
    process_values = list(threads)
    if len(time_values) != len(process_values):
        raise ValueError(
            "Количество времён и чисел процессов должно совпадать"
        )
    if not time_values:
        raise ValueError("Список времён не должен быть пустым")
    if any(not np.isfinite(value) or value <= 0.0 for value in time_values):
        raise ValueError("Все времена должны быть положительными")

    speedup = time_values[0] / np.asarray(time_values, dtype=np.float64)
    positions = np.arange(len(process_values))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(positions, speedup, color="skyblue")
    plt.xticks(positions, [str(value) for value in process_values])
    plt.ylabel("Ускорение")
    plt.xlabel("Количество процессов")
    plt.title(title)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    _save_or_show(save_path)
