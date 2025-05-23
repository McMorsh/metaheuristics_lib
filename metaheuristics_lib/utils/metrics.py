"""
utils/metrics.py

Модуль для вычисления ключевых метрик качества и поведения метаэвристических алгоритмов.
Предназначен для анализа сходимости, стабильности и разнообразия решений.
"""
from typing import List, Sequence, Callable, Optional

import numpy as np


def best_so_far(history: Sequence[float]) -> List[float]:
    """
    По последовательности значений функции стоимости (fitness) за итерации
    возвращает траекторию лучшего (минимального) значения на пройденных итерациях.

    history: [f1, f2, f3, ...]
    returns: [min(f1), min(f1,f2), min(f1,f2,f3), ...]
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


def run_multiple(runner: Callable[..., List[float]], runs: int = 30, *args, **kwargs) -> List[List[float]]:
    """
    Провести несколько запусков алгоритма для статистического анализа.

    :param runner: функция/метод, который возвращает history (list of fitness)
    :param runs: число запусков
    :param args, kwargs: передаются в runner
    :return: список history для каждого запуска
    """
    results = []
    for i in range(runs):  # фиксируем разные seed внутри runner, если нужно
        history = runner(*args, **kwargs)
        results.append(history)
    return results


def summarize_runs(histories: Sequence[Sequence[float]], target: Optional[float] = None) -> dict:
    """
    Собирает ключевые статистики по множеству запусков:
    - mean_final: среднее финальных best
    - std_final: стандартное отклонение финальных best
    - mean_auc: средний AUC
    - time_to_target: среднее время до достижения target (если задан)

    :param histories: список history списков
    :param target: целевое fitness для time_to_target
    :return: словарь со статистиками
    """
    finals = [min(h) for h in histories]
    aucs = [area_under_curve(h, normalize=True) for h in histories]
    summary = {
        'mean_final': float(np.mean(finals)),
        'std_final': float(np.std(finals)),
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'runs': len(histories)
    }
    if target is not None:
        tts = [time_to_target(h, target) for h in histories]
        # фильтруем None
        tts_valid = [t for t in tts if t is not None]
        summary['mean_time_to_target'] = float(np.mean(tts_valid)) if tts_valid else None
        summary['success_rate'] = len(tts_valid) / len(tts)
    return summary
