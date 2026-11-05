"""Тестовые задачи для метаэвристических алгоритмов."""

from metaheuristics_lib.problems.continuous import (
    ackley,
    griewank,
    himmelblau,
    mccormick,
    rastrigin,
    rosenbrock,
    sphere,
    three_hump_camel,
)
from metaheuristics_lib.problems.registry import (
    PROBLEMS,
    ProblemSpec,
    get_problem,
    list_problems,
)

__all__ = [
    "PROBLEMS",
    "ProblemSpec",
    "ackley",
    "get_problem",
    "griewank",
    "himmelblau",
    "list_problems",
    "mccormick",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "three_hump_camel",
]
