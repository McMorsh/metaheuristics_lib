"""
utils/metrics.py

Модуль для вычисления ключевых метрик качества и поведения метаэвристических алгоритмов.
Предназначен для анализа сходимости, стабильности и разнообразия решений.
"""
from typing import List, Sequence, Optional

import numpy as np


def best_so_far(history: Sequence[float]) -> List[float]:
    """
    По последовательности значений функции стоимости (fitness) за итерации
    возвращает траекторию лучшего (минимального) значения на пройденных итерациях.

    :param history: Список fitness значений по итерациям

    :return: [min(f1), min(f1,f2), min(f1,f2,f3), ...]
    """
    best = np.minimum.accumulate(history)
    return best.tolist()


def area_under_curve(history: Sequence[float], normalize: bool = True) -> float:
    """
    Вычисляет площадь под кривой best-so-far по итерациям.
    Чем меньше AUC, тем быстрее алгоритм достиг хорошего решения.

    :param history: Список fitness значений по итерациям
    :param normalize: если True, нормирует AUC на число итераций

    :return: AUC или нормированный AUC
    """
    best = np.minimum.accumulate(history)
    auc = np.trapezoid(best, dx=1)
    if normalize:
        auc = auc / len(history)
    return float(auc)


def time_to_target(history: Sequence[float], target: float) -> Optional[int]:
    """
    Определяет номер итерации, на которой впервые было достигнуто
    значение fitness <= target.

    :param history: Список fitness по итерациям
    :param target: целевое значение fitness

    :return: индекс итерации (0-based) или None, если не достигнуто
    """
    for i, v in enumerate(history):
        if v <= target + 10e-5:
            return i
    return None


def mean_best(history_list: Sequence[Sequence[float]]) -> float:
    """
    Среднее из лучших значений (последний элемент best-so-far)
    по нескольким запускам алгоритма.

    :param history_list: Список списков history для каждого запуска

    :return: среднее финальное best
    """
    finals = [min(h) for h in history_list]
    return float(np.mean(finals))


def run_multiple(runner, runs: int = 30, seed=None):
    """
    Провести несколько запусков алгоритма для статистического анализа.

    :param runner: Данные для запуска алгоритма
    :param runs: число запусков
    :param seed: seed для инициализации генератора случайных чисел.

    :return: 1
    """
    results = []
    list_of_best_fitness = []
    list_of_best_x = []
    list_of_history = []
    list_of_time = []
    list_of_min_x = []

    for i in range(runs):
        if seed is not None:
            runner.seed = seed + i

        best, history_of_best_fitness, history_of_best_x, elapsed_time = runner.run()

        list_of_best_fitness.append(best["minimum_value"])
        list_of_history.append(history_of_best_fitness)

        list_of_best_x.append(best["minimum_x"])
        list_of_min_x.append(history_of_best_x)

        list_of_time.append(elapsed_time)

    results.append({
        "best_fitness": list_of_best_fitness,
        "history_of_best_fitness": list_of_history,
        "best_x": list_of_best_x,
        "history_of_best_x": list_of_min_x[0],
        "time": list_of_time
    })

    return results


def summarize_runs(results: list[dict[str, list[float]]], target: Optional[float] = None) -> dict:
    """
    Собирает ключевые статистики по множеству запусков:
    - mean_final: среднее финальных best
    - std_final: стандартное отклонение финальных best
    - mean_auc: средний AUC
    - std_auc: стандартное отклонение AUC
    - time_to_target: среднее время до достижения target (если задан)
    - time: среднее время работы алгоритма

    :param results: результаты запусков алгоритмов (histories, bests,times)
    :param target: целевое fitness для time_to_target

    :return: словарь со статистиками
    """

    aucs = [area_under_curve(h, normalize=True) for h in results[0]["history_of_best_fitness"]]

    summary = {
        'mean_final': float(np.mean(results[0]["best_fitness"])),
        'std_final': float(np.std(results[0]["best_fitness"])),
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'time': float(np.mean(results[0]["time"])),
    }

    if target is not None:
        tts = [time_to_target(h, target) for h in results[0]["history_of_best_fitness"]]
        # фильтруем None
        tts_valid = [t for t in tts if t is not None]
        summary['mean_time_to_target'] = float(np.mean(tts_valid)) if tts_valid else None
        summary['success_rate'] = len(tts_valid) / len(tts)

    return summary
