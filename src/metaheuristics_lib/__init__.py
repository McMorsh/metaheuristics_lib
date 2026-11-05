"""Публичный фасад библиотеки метаэвристик.

Пакет реэкспортирует стабильный API из подпакетов ``core``,
``algorithms`` и ``problems``. Внутренние модули импортируются
через ``metaheuristics_lib.*``.
"""

from metaheuristics_lib.algorithms import (
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
)
from metaheuristics_lib.core import (
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
from metaheuristics_lib.problems import (
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
