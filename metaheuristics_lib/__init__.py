"""Публичный фасад библиотеки метаэвристик.

Пакет реэкспортирует стабильный API из плоских пакетов ``core``,
``algorithms`` и ``problems``. Внутренние импорты проекта по-прежнему
используют плоские имена, чтобы не ломать существующие примеры.
"""

from algorithms import (
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
)
from core import (
    BaseAlgorithm,
    BoundsLike,
    Evaluator,
    MPIEvaluator,
    Objective,
    OptimizationResult,
    ProcessPoolEvaluator,
    Runner,
    SerialEvaluator,
    normalize_bounds,
)
from problems import (
    ackley,
    griewank,
    himmelblau,
    mccormick,
    rastrigin,
    rosenbrock,
    sphere,
    three_hump_camel,
)

__all__ = [
    "ArtificialBeeColony",
    "BaseAlgorithm",
    "BoundsLike",
    "CrowSearchAlgorithm",
    "EmperorPenguinOptimizer",
    "Evaluator",
    "GreyWolfOptimizer",
    "MPIEvaluator",
    "Objective",
    "OptimizationResult",
    "ProcessPoolEvaluator",
    "Runner",
    "SerialEvaluator",
    "WhaleOptimizationAlgorithm",
    "ackley",
    "griewank",
    "himmelblau",
    "mccormick",
    "normalize_bounds",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "three_hump_camel",
]
