"""Сравнение алгоритмов по качеству, времени и числу обращений."""

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from metaheuristics_lib import (
    ArtificialBeeColony,
    BaseAlgorithm,
    BoundsLike,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    Objective,
    Runner,
    WhaleOptimizationAlgorithm,
)
from metaheuristics_lib.problems.registry import get_problem, list_problems

AlgorithmFactory = Callable[..., BaseAlgorithm]

ALGORITHMS: list[AlgorithmFactory] = [
    ArtificialBeeColony,
    CrowSearchAlgorithm,
    EmperorPenguinOptimizer,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
]


@dataclass(frozen=True)
class BenchmarkRow:
    """Строка результата одного алгоритма.

    Attributes:
        name: имя класса алгоритма.
        best: среднее лучшее значение функции по запускам.
        nfev: среднее число обращений к функции за запуск.
        elapsed: среднее время запуска в секундах.
    """

    name: str
    best: float
    nfev: int
    elapsed: float


def run_benchmark(
    problem: Objective,
    bounds: BoundsLike,
    dim: int,
    iterations: int,
    agents: int,
    runs: int,
    seed: int,
) -> list[BenchmarkRow]:
    """Запускает все алгоритмы и собирает средние показатели.

    Args:
        problem: целевая функция.
        bounds: границы поиска.
        dim: размерность задачи.
        iterations: число итераций каждого запуска.
        agents: размер популяции.
        runs: число запусков на алгоритм.
        seed: базовый seed; запуск ``i`` получает ``seed + i``.

    Returns:
        Строки результатов в порядке :data:`ALGORITHMS`.

    Raises:
        ValueError: если ``runs`` меньше единицы.
    """
    if runs < 1:
        raise ValueError(
            f"Параметр runs должен быть положительным, получено {runs}"
        )

    rows: list[BenchmarkRow] = []
    for factory in ALGORITHMS:
        values: list[float] = []
        counts: list[int] = []
        times: list[float] = []
        for run in range(runs):
            algorithm = factory(
                problem,
                dim,
                bounds,
                agents=agents,
                max_iter=iterations,
                seed=seed + run,
            )
            with Runner(algorithm, iteration_info=False) as runner:
                result, _, _, elapsed = runner.run()
            values.append(result["minimum_value"])
            counts.append(algorithm.nfev)
            times.append(elapsed)

        rows.append(
            BenchmarkRow(
                name=factory.__name__,
                best=statistics.fmean(values),
                nfev=int(statistics.fmean(counts)),
                elapsed=statistics.fmean(times),
            )
        )
    return rows


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Форматирует результаты в таблицу для печати."""
    header = f"{'Алгоритм':32s} {'f(x)':>12s} {'nfev':>10s} {'время, с':>10s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.name:32s} {row.best:12.4e} "
            f"{row.nfev:10d} {row.elapsed:10.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Разбирает аргументы и печатает таблицу сравнения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        choices=list_problems(),
        default="rastrigin",
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problem = get_problem(args.problem)
    if problem.dim is not None and args.dim != problem.dim:
        parser.error(f"Задача {problem.name} определена для dim={problem.dim}")
    dim = problem.dim if problem.dim is not None else args.dim
    rows = run_benchmark(
        problem.function,
        problem.bounds,
        dim,
        args.iterations,
        args.agents,
        args.runs,
        args.seed,
    )
    print(format_rows(rows))


if __name__ == "__main__":
    main()
