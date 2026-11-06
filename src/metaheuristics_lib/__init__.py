"""Публичный фасад библиотеки метаэвристик.

Пакет реэкспортирует стабильный API из подпакетов ``core``,
``algorithms`` и ``problems``. Внутренние модули импортируются
через ``metaheuristics_lib.*``.
"""

from importlib.metadata import PackageNotFoundError, version

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
    PROBLEMS,
    ProblemSpec,
    ackley,
    get_problem,
    griewank,
    himmelblau,
    list_problems,
    mccormick,
    rastrigin,
    rosenbrock,
    sphere,
    three_hump_camel,
)

try:
    __version__ = version("metaheuristics-lib")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "PROBLEMS",
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
    "ProblemSpec",
    "ProcessPoolEvaluator",
    "Runner",
    "SerialEvaluator",
    "WhaleOptimizationAlgorithm",
    "__version__",
    "ackley",
    "get_problem",
    "griewank",
    "himmelblau",
    "list_problems",
    "mccormick",
    "normalize_bounds",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "three_hump_camel",
]
