"""Метрики сходимости и статистика запусков."""

import warnings
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from metaheuristics_lib.core.runner import Runner

type History = Sequence[float] | npt.NDArray[np.float64]
type HistoryList = Sequence[History] | npt.NDArray[np.float64]


class MultipleRunResults(TypedDict):
    """Результаты нескольких запусков алгоритма.

    Attributes:
        best_fitness: лучшее значение в каждом запуске.
        history_of_best_fitness: истории лучшего значения.
        best_x: лучшие координаты каждого запуска.
        history_of_best_x: история лучших координат каждого запуска.
        time: время каждого запуска в секундах.
    """

    best_fitness: list[float]
    history_of_best_fitness: list[list[float]]
    best_x: list[list[float]]
    history_of_best_x: list[list[list[float]]]
    time: list[float]


def best_so_far(history: History) -> list[float]:
    """Возвращает траекторию лучшего значения функции.

    Args:
        history: значения функции по итерациям.

    Returns:
        Последовательность накопленных минимумов.

    Raises:
        ValueError: если история не является одномерной.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("История должна быть одномерной")
    best = np.minimum.accumulate(values)
    return [float(value) for value in best]


def area_under_curve(history: History, normalize: bool = True) -> float:
    """Вычисляет площадь под кривой best-so-far.

    Args:
        history: значения функции по итерациям.
        normalize: делить ли площадь на число итераций.

    Returns:
        Площадь под кривой.

    Raises:
        ValueError: если история пуста.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("История должна быть одномерной")
    if values.size == 0:
        raise ValueError("История не должна быть пустой")

    best = np.minimum.accumulate(values)
    auc = float(np.trapezoid(best, dx=1.0))
    if normalize:
        auc /= values.size
    return auc


def time_to_target(history: History, target: float) -> int | None:
    """Находит первую итерацию с достижением цели.

    Args:
        history: значения функции по итерациям.
        target: целевое значение функции.

    Returns:
        Индекс итерации (с нуля) или ``None``, если цель не достигнута.

    Raises:
        ValueError: если история не является одномерной.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("История должна быть одномерной")
    for index, value in enumerate(values):
        if value <= target + 1e-4:
            return index
    return None


def mean_best(history_list: HistoryList) -> float:
    """Возвращает среднее финальных лучших значений.

    Args:
        history_list: истории нескольких запусков.

    Returns:
        Среднее значение лучших результатов.

    Raises:
        ValueError: если список историй пуст или содержит пустую историю.
    """
    histories = list(history_list)
    if not histories:
        raise ValueError("Список историй не должен быть пустым")

    finals: list[float] = []
    for history in histories:
        values = np.asarray(history, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("История должна быть одномерной")
        if values.size == 0:
            raise ValueError("История не должна быть пустой")
        finals.append(float(np.min(values)))
    return float(np.mean(finals))


def run_repeated(
    runner: Runner,
    runs: int = 30,
    seed: int | None = None,
) -> MultipleRunResults:
    """Выполняет несколько запусков алгоритма.

    Args:
        runner: обёртка запуска. Для пула процессов передавайте её
            внутри ``with``, чтобы пул не пересоздавался между
            запусками.
        runs: число запусков, не меньше 1.
        seed: базовый seed. Запуск ``i`` получает ``seed + i``. Если
            параметр не задан, используется seed раннера или алгоритма.

    Returns:
        Словарь с массивами результатов по запускам.

    Raises:
        ValueError: если ``runs < 1``.
    """
    if runs < 1:
        raise ValueError(
            f"Параметр runs должен быть положительным, получено {runs}"
        )

    base_seed = seed
    if base_seed is None:
        base_seed = runner.seed
    if base_seed is None:
        base_seed = runner.algorithm.seed

    best_fitness: list[float] = []
    history_of_best_fitness: list[list[float]] = []
    best_x: list[list[float]] = []
    history_of_best_x: list[list[list[float]]] = []
    elapsed: list[float] = []

    original_runner_seed = runner.seed
    original_algorithm_seed = runner.algorithm.seed
    try:
        for index in range(runs):
            if base_seed is not None:
                runner.seed = base_seed + index

            result, fitness_history, x_history, run_time = runner.run()

            best_fitness.append(result["minimum_value"])
            history_of_best_fitness.append(fitness_history)
            best_x.append(result["minimum_x"])
            history_of_best_x.append(x_history)
            elapsed.append(run_time)
    finally:
        runner.seed = original_runner_seed
        runner.algorithm.set_seed(original_algorithm_seed)

    return MultipleRunResults(
        best_fitness=best_fitness,
        history_of_best_fitness=history_of_best_fitness,
        best_x=best_x,
        history_of_best_x=history_of_best_x,
        time=elapsed,
    )


def run_multiple(
    runner: Runner,
    runs: int = 30,
    seed: int | None = None,
) -> list[MultipleRunResults]:
    """Устаревшая обёртка над :func:`run_repeated`.

    Возвращает список из одного словаря для совместимости со старым
    форматом. Новый код должен использовать :func:`run_repeated`.

    Args:
        runner: обёртка запуска.
        runs: число запусков, не меньше 1.
        seed: базовый seed.

    Returns:
        Список из одного словаря с результатами.

    Raises:
        ValueError: если ``runs < 1``.
    """
    warnings.warn(
        "run_multiple устарела; используйте run_repeated",
        DeprecationWarning,
        stacklevel=2,
    )
    return [run_repeated(runner, runs, seed)]


def summarize_runs(
    results: Sequence[MultipleRunResults],
    target: float | None = None,
) -> dict[str, float | None]:
    """Собирает статистику по результатам запусков.

    Args:
        results: список результатов :func:`run_repeated`.
        target: целевое значение для ``time_to_target``.

    Returns:
        Словарь со средними значениями, стандартными отклонениями и
        долей успешных запусков.

    Raises:
        ValueError: если результаты или истории пусты.
    """
    if not results:
        raise ValueError("Список результатов не должен быть пустым")

    first = results[0]
    histories = first["history_of_best_fitness"]
    if not histories:
        raise ValueError("Истории запусков не должны быть пустыми")

    aucs = [area_under_curve(history) for history in histories]
    summary: dict[str, float | None] = {
        "mean_final": float(np.mean(first["best_fitness"])),
        "std_final": float(np.std(first["best_fitness"])),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "time": float(np.mean(first["time"])),
    }

    if target is not None:
        times = [time_to_target(history, target) for history in histories]
        valid_times = [value for value in times if value is not None]
        summary["mean_time_to_target"] = (
            float(np.mean(valid_times)) if valid_times else None
        )
        summary["success_rate"] = len(valid_times) / len(times)

    return summary


def summarize_run(
    result: MultipleRunResults,
    target: float | None = None,
) -> dict[str, float | None]:
    """Собирает статистику по результату :func:`run_repeated`.

    Args:
        result: словарь с результатами запусков.
        target: целевое значение для ``time_to_target``.

    Returns:
        Словарь со средними значениями, стандартными отклонениями и
        долей успешных запусков.
    """
    return summarize_runs([result], target=target)
