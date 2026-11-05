"""Реестр тестовых задач с границами и известными оптимумами."""

from dataclasses import dataclass

from metaheuristics_lib.core.types import BoundsLike, Objective
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


@dataclass(frozen=True)
class ProblemSpec:
    """Описание тестовой задачи.

    Attributes:
        name: короткое имя для поиска и командной строки.
        function: целевая функция.
        dim: фиксированная размерность или ``None`` для любой
            положительной размерности.
        bounds: общие границы или интервалы по размерностям.
        optimum: известное минимальное значение.
        optimum_x: координаты одного оптимума для фиксированной
            размерности; ``None``, если точка зависит от размерности.
    """

    name: str
    function: Objective
    dim: int | None
    bounds: BoundsLike
    optimum: float
    optimum_x: tuple[float, ...] | None = None


PROBLEMS: dict[str, ProblemSpec] = {
    "ackley": ProblemSpec(
        name="ackley",
        function=ackley,
        dim=None,
        bounds=(-32.768, 32.768),
        optimum=0.0,
    ),
    "griewank": ProblemSpec(
        name="griewank",
        function=griewank,
        dim=None,
        bounds=(-600.0, 600.0),
        optimum=0.0,
    ),
    "himmelblau": ProblemSpec(
        name="himmelblau",
        function=himmelblau,
        dim=2,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        optimum_x=(3.0, 2.0),
    ),
    "mccormick": ProblemSpec(
        name="mccormick",
        function=mccormick,
        dim=2,
        bounds=((-1.5, 4.0), (-3.0, 4.0)),
        optimum=-1.913222954882274,
        optimum_x=(-0.54719, -1.54719),
    ),
    "rastrigin": ProblemSpec(
        name="rastrigin",
        function=rastrigin,
        dim=None,
        bounds=(-5.12, 5.12),
        optimum=0.0,
    ),
    "rosenbrock": ProblemSpec(
        name="rosenbrock",
        function=rosenbrock,
        dim=None,
        bounds=(-5.0, 10.0),
        optimum=0.0,
    ),
    "sphere": ProblemSpec(
        name="sphere",
        function=sphere,
        dim=None,
        bounds=(-5.12, 5.12),
        optimum=0.0,
    ),
    "three_hump_camel": ProblemSpec(
        name="three_hump_camel",
        function=three_hump_camel,
        dim=2,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        optimum_x=(0.0, 0.0),
    ),
}


def get_problem(name: str) -> ProblemSpec:
    """Возвращает задачу по имени.

    Args:
        name: имя из :data:`PROBLEMS`.

    Returns:
        Описание задачи.

    Raises:
        KeyError: если задача с таким именем не зарегистрирована.
    """
    try:
        return PROBLEMS[name]
    except KeyError as error:
        available = ", ".join(list_problems())
        raise KeyError(
            f"Неизвестная задача: {name}. Доступны: {available}"
        ) from error


def list_problems() -> list[str]:
    """Возвращает имена зарегистрированных задач по алфавиту."""
    return sorted(PROBLEMS)
