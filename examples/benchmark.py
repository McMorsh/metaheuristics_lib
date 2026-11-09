"""Сравнение алгоритмов по качеству, времени и числу обращений."""

import argparse
import logging
import statistics
from collections import Counter
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
from metaheuristics_lib.utils.metrics import best_at_budget

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
        stop_reason: распределение причин остановки по запускам или
            ``None``, если все запуски дошли до ``max_iter``.
        best_at_budget: среднее лучшее значение при заданном бюджете
            или ``None``, если бюджет не задан.
        budget_reached: число запусков, дошедших до бюджета, в виде
            ``"2/3"`` или ``None``, если бюджет не задан.
    """

    name: str
    best: float
    nfev: int
    elapsed: float
    stop_reason: str | None = None
    best_at_budget: float | None = None
    budget_reached: str | None = None


def _summarize_stop_reasons(
    reasons: list[str | None],
) -> str | None:
    """Сводит причины остановки запусков в одну строку.

    Args:
        reasons: причины остановки по запускам.

    Returns:
        Распределение причин по убыванию частоты в виде
        ``"target 2/3, max_fev 1/3"`` или ``None``, если ранней
        остановки не было.
    """
    counts = Counter(reason for reason in reasons if reason is not None)
    if not counts:
        return None

    total = len(reasons)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{reason} {count}/{total}" for reason, count in ordered)


def run_benchmark(
    problem: Objective,
    bounds: BoundsLike,
    dim: int,
    iterations: int,
    agents: int,
    runs: int,
    seed: int,
    *,
    max_fev: int | None = None,
    target: float | None = None,
    budget: int | None = None,
    iteration_info: bool | int = False,
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
        max_fev: мягкий лимит обращений к целевой функции.
        target: целевое значение fitness для ранней остановки.
        budget: бюджет обращений к функции для сравнения. Если
            ``max_fev`` не задан, он устанавливается равным бюджету.
        iteration_info: настройка логов итераций для :class:`Runner`.

    Returns:
        Строки результатов в порядке :data:`ALGORITHMS`.

    Raises:
        ValueError: если ``runs`` меньше единицы.
    """
    if runs < 1:
        raise ValueError(
            f"Параметр runs должен быть положительным, получено {runs}"
        )
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError(
                f"Параметр budget должен быть целым, получено {budget!r}"
            )
        if budget < 1:
            raise ValueError(
                f"Параметр budget должен быть положительным, получено {budget}"
            )

    effective_max_fev = max_fev
    if budget is not None and effective_max_fev is None:
        effective_max_fev = budget

    rows: list[BenchmarkRow] = []
    for factory in ALGORITHMS:
        values: list[float] = []
        counts: list[int] = []
        times: list[float] = []
        reasons: list[str | None] = []
        budget_values: list[float] = []
        budget_hits = 0
        for run in range(runs):
            algorithm = factory(
                problem,
                dim,
                bounds,
                agents=agents,
                max_iter=iterations,
                seed=seed + run,
                max_fev=effective_max_fev,
                target=target,
            )
            with Runner(
                algorithm,
                iteration_info=iteration_info,
            ) as runner:
                result, _, _, elapsed = runner.run()
            values.append(result["minimum_value"])
            counts.append(algorithm.nfev)
            times.append(elapsed)
            reasons.append(algorithm.stop_reason)
            if budget is not None and algorithm.nfev >= budget:
                value = best_at_budget(
                    algorithm.nfev_history,
                    algorithm.best_fitness_history,
                    budget,
                )
                if value is not None:
                    budget_values.append(value)
                    budget_hits += 1

        rows.append(
            BenchmarkRow(
                name=factory.__name__,
                best=statistics.fmean(values),
                nfev=int(statistics.fmean(counts)),
                elapsed=statistics.fmean(times),
                stop_reason=_summarize_stop_reasons(reasons),
                best_at_budget=(
                    statistics.fmean(budget_values) if budget_values else None
                ),
                budget_reached=(
                    f"{budget_hits}/{runs}" if budget is not None else None
                ),
            )
        )
    return rows


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Форматирует результаты в таблицу для печати."""
    stop_width = max(
        len("остановка"),
        max((len(row.stop_reason or "-") for row in rows), default=0),
    )
    has_budget = any(row.budget_reached is not None for row in rows)
    header = (
        f"{'Алгоритм':32s} {'f(x)':>12s} {'nfev':>10s} "
        f"{'время, с':>10s} {'остановка':>{stop_width}s}"
    )
    if has_budget:
        header += f" {'f(x)@budget':>12s} {'budget':>8s}"

    lines = [header, "-" * len(header)]
    for row in rows:
        stop_reason = row.stop_reason or "-"
        line = (
            f"{row.name:32s} {row.best:12.4e} "
            f"{row.nfev:10d} {row.elapsed:10.3f} "
            f"{stop_reason:>{stop_width}s}"
        )
        if has_budget:
            budget_value = (
                f"{row.best_at_budget:12.4e}"
                if row.best_at_budget is not None
                else f"{'-':>12s}"
            )
            budget_reached = row.budget_reached or "-"
            line += f" {budget_value} {budget_reached:>8s}"
        lines.append(line)
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
    parser.add_argument(
        "--max-fev",
        type=int,
        default=None,
        help="мягкий лимит обращений к целевой функции",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="целевое значение fitness для ранней остановки",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="бюджет обращений к функции для сравнения",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="логировать итерации алгоритмов",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

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
        max_fev=args.max_fev,
        target=args.target,
        budget=args.budget,
        iteration_info=args.verbose,
    )
    print(format_rows(rows))


if __name__ == "__main__":
    main()
