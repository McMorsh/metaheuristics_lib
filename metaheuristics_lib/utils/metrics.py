"""
utils/metrics.py

Модуль для вычисления ключевых метрик качества и поведения метаэвристических алгоритмов.
Предназначен для анализа сходимости, стабильности и разнообразия решений.
"""
from typing import List, Sequence, Callable, Optional, Any

import numpy as np

from core.runner import Runner


def best_so_far(history: Sequence[float]) -> List[float]:
    """
    По последовательности значений функции стоимости (fitness) за итерации
    возвращает траекторию лучшего (минимального) значения на пройденных итерациях.

    :param history: список fitness значений по итерациям

    :return: [min(f1), min(f1,f2), min(f1,f2,f3), ...]
    """
    best = np.minimum.accumulate(history)
    return best.tolist()


def area_under_curve(history: Sequence[float], normalize: bool = True) -> float:
    """
    Вычисляет площадь под кривой best-so-far по итерациям.
    Чем меньше AUC, тем быстрее алгоритм достиг хорошего решения.

    :param history: список fitness значений по итерациям
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

    :param history: список fitness по итерациям
    :param target: целевое значение fitness

    :return: индекс итерации (0-based) или None, если не достигнуто
    """
    for i, v in enumerate(history):
        if v <= target:
            return i
    return None


def mean_best(history_list: Sequence[Sequence[float]]) -> float:
    """
    Среднее из лучших значений (последний элемент best-so-far)
    по нескольким запускам алгоритма.

    :param history_list: список списков history для каждого запуска

    :return: среднее финальное best
    """
    finals = [min(h) for h in history_list]
    return float(np.mean(finals))


def run_multiple(runner, runs: int = 30, seed = None) -> List[List[float]]:
    """
    Провести несколько запусков алгоритма для статистического анализа.

    :param algo: 1
    :param runs: число запусков

    :return: 1
    """
    results = []
    list_of_best = []
    list_of_history = []
    list_of_time = []

    for i in range(runs):
        if seed is not None:
            runner.seed = seed + i

        best, history, elapsed_time = runner.run()
        list_of_best.append(best["minimum_value"])
        list_of_history.append(history)
        list_of_time.append(elapsed_time)

    results.append({
        "best": list_of_best,
        "history": list_of_history,
        "time": list_of_time
    })

    return results


def summarize_runs(results: list[dict[str, list[float]]], target: Optional[float] = None) -> dict:
    """
    Собирает ключевые статистики по множеству запусков:
    - mean_final: среднее финальных best
    - std_final: стандартное отклонение финальных best
    - mean_auc: средний AUC
    - time_to_target: среднее время до достижения target (если задан)``

    :param results
    :param target: целевое fitness для time_to_target

    :return: словарь со статистиками
    """
    aucs = [area_under_curve(h, normalize=True) for h in results[0]["history"]]
    summary = {
        'mean_final': float(np.mean(results[0]["best"])),
        'std_final': float(np.std(results[0]["best"])),
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'time': float(np.mean(results[0]["time"])),
    }
    if target is not None:
        tts = [time_to_target(h, target) for h in results[0]["history"]]
        # фильтруем None
        tts_valid = [t for t in tts if t is not None]
        summary['mean_time_to_target'] = float(np.mean(tts_valid)) if tts_valid else None
        summary['success_rate'] = len(tts_valid) / len(tts)

    return summary
