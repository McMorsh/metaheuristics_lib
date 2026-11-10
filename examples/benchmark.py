"""Сравнение алгоритмов по качеству, времени и числу обращений."""

import argparse
import csv
import json
import logging
import math
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

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

RUN_FIELDS = (
    "algorithm",
    "problem",
    "dim",
    "seed",
    "best",
    "nfev",
    "stop_reason",
    "elapsed",
    "budget",
    "best_at_budget",
)


@dataclass(frozen=True)
class RunRecord:
    """Сырой результат одного запуска.

    Attributes:
        algorithm: имя класса алгоритма.
        problem: имя задачи.
        dim: размерность задачи.
        seed: seed запуска.
        best: лучшее значение функции.
        nfev: число обращений к функции.
        stop_reason: причина остановки или ``None``.
        elapsed: время запуска в секундах.
        budget: бюджет сравнения или ``None``.
        best_at_budget: лучшее значение на бюджете или ``None``.
    """

    algorithm: str
    problem: str
    dim: int
    seed: int
    best: float
    nfev: int
    stop_reason: str | None
    elapsed: float
    budget: int | None
    best_at_budget: float | None


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
        problem: имя задачи в сводной таблице или ``None``.
        dim: размерность задачи в сводной таблице или ``None``.
        median: медиана лучшего значения по запускам.
        std: стандартное отклонение лучшего значения по запускам.
        success_rate: доля запусков, достигших ``target``.
        runs: число запусков в строке.
    """

    name: str
    best: float
    nfev: int
    elapsed: float
    stop_reason: str | None = None
    best_at_budget: float | None = None
    budget_reached: str | None = None
    problem: str | None = None
    dim: int | None = None
    median: float | None = None
    std: float | None = None
    success_rate: float | None = None
    runs: int = 0


def _validate_runs_and_budget(
    runs: int,
    budget: int | None,
) -> None:
    """Проверяет число запусков и бюджет."""
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


def _effective_max_fev(
    max_fev: int | None,
    budget: int | None,
) -> int | None:
    """Выбирает лимит вычислений с учётом бюджета."""
    if budget is not None and max_fev is None:
        return budget
    return max_fev


def _is_target_reached(value: float, target: float) -> bool:
    """Проверяет достижение цели с допуском."""
    return value <= target or math.isclose(
        value,
        target,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def _run_one(
    factory: AlgorithmFactory,
    problem: Objective,
    bounds: BoundsLike,
    dim: int,
    iterations: int,
    agents: int,
    seed: int,
    problem_name: str,
    *,
    max_fev: int | None,
    target: float | None,
    budget: int | None,
    iteration_info: bool | int,
) -> RunRecord:
    """Выполняет один запуск и возвращает сырой результат."""
    effective_max_fev = _effective_max_fev(max_fev, budget)
    algorithm = factory(
        problem,
        dim,
        bounds,
        agents=agents,
        max_iter=iterations,
        seed=seed,
        max_fev=effective_max_fev,
        target=target,
    )
    with Runner(
        algorithm,
        iteration_info=iteration_info,
    ) as runner:
        result, _, _, elapsed = runner.run()

    value_at_budget = None
    if budget is not None and algorithm.nfev >= budget:
        value_at_budget = best_at_budget(
            algorithm.nfev_history,
            algorithm.best_fitness_history,
            budget,
        )

    return RunRecord(
        algorithm=factory.__name__,
        problem=problem_name,
        dim=dim,
        seed=seed,
        best=result["minimum_value"],
        nfev=algorithm.nfev,
        stop_reason=algorithm.stop_reason,
        elapsed=elapsed,
        budget=budget,
        best_at_budget=value_at_budget,
    )


def collect_runs(
    problem: Objective,
    bounds: BoundsLike,
    dim: int,
    iterations: int,
    agents: int,
    runs: int,
    seed: int,
    *,
    problem_name: str,
    max_fev: int | None = None,
    target: float | None = None,
    budget: int | None = None,
    iteration_info: bool | int = False,
) -> list[RunRecord]:
    """Запускает все алгоритмы и возвращает сырые результаты.

    Args:
        problem: целевая функция.
        bounds: границы поиска.
        dim: размерность задачи.
        iterations: число итераций каждого запуска.
        agents: размер популяции.
        runs: число запусков на алгоритм.
        seed: базовый seed; запуск ``i`` получает ``seed + i``.
        problem_name: имя задачи для записи.
        max_fev: мягкий лимит обращений к целевой функции.
        target: целевое значение fitness для ранней остановки.
        budget: бюджет обращений к функции для сравнения.
        iteration_info: настройка логов итераций для :class:`Runner`.

    Returns:
        Записи в порядке алгоритмов и запусков.

    Raises:
        TypeError: если бюджет не является целым числом.
        ValueError: если ``runs`` или ``budget`` меньше единицы.
    """
    _validate_runs_and_budget(runs, budget)

    records: list[RunRecord] = []
    for factory in ALGORITHMS:
        for run in range(runs):
            records.append(
                _run_one(
                    factory,
                    problem,
                    bounds,
                    dim,
                    iterations,
                    agents,
                    seed + run,
                    problem_name,
                    max_fev=max_fev,
                    target=target,
                    budget=budget,
                    iteration_info=iteration_info,
                )
            )
    return records


def run_experiments(
    problem_names: list[str],
    dims: list[int],
    iterations: int,
    agents: int,
    runs: int,
    seed: int,
    *,
    max_fev: int | None = None,
    target: float | None = None,
    budget: int | None = None,
    iteration_info: bool | int = False,
) -> list[RunRecord]:
    """Запускает эксперименты по нескольким задачам и размерностям.

    Args:
        problem_names: имена задач из реестра.
        dims: размерности для задач с переменной размерностью.
        iterations: число итераций каждого запуска.
        agents: размер популяции.
        runs: число запусков на алгоритм.
        seed: базовый seed.
        max_fev: мягкий лимит обращений к целевой функции.
        target: целевое значение fitness для ранней остановки.
        budget: бюджет обращений к функции для сравнения.
        iteration_info: настройка логов итераций для :class:`Runner`.

    Returns:
        Записи по всем задачам, размерностям, алгоритмам и запускам.

    Raises:
        ValueError: если списки задач или размерностей пусты либо
            фиксированная размерность задачи не совпадает с ``dims``.
    """
    if not problem_names:
        raise ValueError("Список задач не должен быть пустым")
    if not dims:
        raise ValueError("Список размерностей не должен быть пустым")

    for name in problem_names:
        spec = get_problem(name)
        if spec.dim is not None and (len(dims) != 1 or dims[0] != spec.dim):
            raise ValueError(f"Задача {name} определена для dim={spec.dim}")

    records: list[RunRecord] = []
    for name in problem_names:
        spec = get_problem(name)
        problem_dims = [spec.dim] if spec.dim is not None else dims
        for dim in problem_dims:
            records.extend(
                collect_runs(
                    spec.function,
                    spec.bounds,
                    dim,
                    iterations,
                    agents,
                    runs,
                    seed,
                    problem_name=name,
                    max_fev=max_fev,
                    target=target,
                    budget=budget,
                    iteration_info=iteration_info,
                )
            )
    return records


def summarize_records(
    records: list[RunRecord],
    *,
    target: float | None = None,
    include_problem: bool = True,
) -> list[BenchmarkRow]:
    """Собирает статистику по сырым результатам.

    Args:
        records: записи запусков.
        target: целевое значение для success rate.
        include_problem: добавлять ли имя задачи и размерность в
            строки результата.

    Returns:
        Строки в порядке первого появления задачи, размерности и
        алгоритма.

    Raises:
        ValueError: если список записей пуст.
    """
    if not records:
        raise ValueError("Список результатов не должен быть пустым")

    grouped: dict[tuple[str | None, int | None, str], list[RunRecord]] = {}
    order: list[tuple[str | None, int | None, str]] = []
    for record in records:
        key = (
            record.problem if include_problem else None,
            record.dim if include_problem else None,
            record.algorithm,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record)

    rows: list[BenchmarkRow] = []
    for key in order:
        problem, dim, algorithm = key
        group = grouped[key]
        values = [record.best for record in group]
        counts = [record.nfev for record in group]
        times = [record.elapsed for record in group]
        budget_values = [
            record.best_at_budget
            for record in group
            if record.best_at_budget is not None
        ]
        budget_requested = any(record.budget is not None for record in group)
        success_rate = None
        if target is not None:
            successes = sum(
                1
                for record in group
                if _is_target_reached(record.best, target)
            )
            success_rate = successes / len(group)

        rows.append(
            BenchmarkRow(
                name=algorithm,
                best=statistics.fmean(values),
                nfev=int(statistics.fmean(counts)),
                elapsed=statistics.fmean(times),
                stop_reason=_summarize_stop_reasons(
                    [record.stop_reason for record in group]
                ),
                best_at_budget=(
                    statistics.fmean(budget_values) if budget_values else None
                ),
                budget_reached=(
                    f"{len(budget_values)}/{len(group)}"
                    if budget_requested
                    else None
                ),
                problem=problem,
                dim=dim,
                median=statistics.median(values),
                std=(statistics.stdev(values) if len(values) > 1 else 0.0),
                success_rate=success_rate,
                runs=len(group),
            )
        )
    return rows


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
        TypeError: если бюджет не является целым числом.
        ValueError: если ``runs`` или ``budget`` меньше единицы.
    """
    records = collect_runs(
        problem,
        bounds,
        dim,
        iterations,
        agents,
        runs,
        seed,
        problem_name=getattr(problem, "__name__", "objective"),
        max_fev=max_fev,
        target=target,
        budget=budget,
        iteration_info=iteration_info,
    )
    return summarize_records(
        records,
        target=target,
        include_problem=False,
    )


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


def save_records(records: list[RunRecord], path: Path) -> None:
    """Сохраняет сырые результаты в JSON или CSV.

    Args:
        records: записи запусков.
        path: путь с расширением ``.json`` или ``.csv``.

    Raises:
        ValueError: если расширение не поддерживается.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = [asdict(record) for record in records]
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    if suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        return

    raise ValueError(
        f"Неподдерживаемый формат: {path.suffix}. Используйте .json или .csv"
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Форматирует результаты в таблицу для печати."""
    stop_width = max(
        len("остановка"),
        max((len(row.stop_reason or "-") for row in rows), default=0),
    )
    problem_width = max(
        len("Задача"),
        max((len(row.problem or "") for row in rows), default=0),
    )
    has_problem = any(row.problem is not None for row in rows)
    has_stats = any(row.runs > 1 for row in rows)
    has_budget = any(row.budget_reached is not None for row in rows)
    has_success = any(row.success_rate is not None for row in rows)

    header_parts: list[str] = []
    if has_problem:
        header_parts.append(f"{'Задача':{problem_width}s}")
        header_parts.append(f"{'dim':>4s}")
    header_parts.append(f"{'Алгоритм':32s}")
    header_parts.append(f"{'f(x)':>12s}")
    if has_stats:
        header_parts.append(f"{'медиана':>12s}")
        header_parts.append(f"{'std':>10s}")
    header_parts.append(f"{'nfev':>10s}")
    header_parts.append(f"{'время, с':>10s}")
    header_parts.append(f"{'остановка':>{stop_width}s}")
    if has_budget:
        header_parts.append(f"{'f(x)@budget':>12s}")
        header_parts.append(f"{'budget':>8s}")
    if has_success:
        header_parts.append(f"{'success':>8s}")
    header = " ".join(header_parts)

    lines = [header, "-" * len(header)]
    for row in rows:
        parts: list[str] = []
        if has_problem:
            parts.append(f"{(row.problem or '-'):{problem_width}s}")
            dim_text = "-" if row.dim is None else str(row.dim)
            parts.append(f"{dim_text:>4s}")
        parts.append(f"{row.name:32s}")
        parts.append(f"{row.best:12.4e}")
        if has_stats:
            median = (
                f"{row.median:12.4e}"
                if row.median is not None
                else f"{'-':>12s}"
            )
            std = f"{row.std:10.4e}" if row.std is not None else f"{'-':>10s}"
            parts.append(median)
            parts.append(std)
        parts.append(f"{row.nfev:10d}")
        parts.append(f"{row.elapsed:10.3f}")
        parts.append(f"{(row.stop_reason or '-'):>{stop_width}s}")
        if has_budget:
            budget_value = (
                f"{row.best_at_budget:12.4e}"
                if row.best_at_budget is not None
                else f"{'-':>12s}"
            )
            budget_reached = row.budget_reached or "-"
            parts.append(budget_value)
            parts.append(f"{budget_reached:>8s}")
        if has_success:
            success = (
                f"{row.success_rate:8.2f}"
                if row.success_rate is not None
                else f"{'-':>8s}"
            )
            parts.append(success)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def main() -> None:
    """Разбирает аргументы и печатает таблицу сравнения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        action="append",
        choices=list_problems(),
        default=None,
        help="имя задачи из реестра; можно указывать несколько раз",
    )
    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="запустить все задачи из реестра",
    )
    parser.add_argument(
        "--dim",
        action="append",
        type=int,
        default=None,
        help="размерность задачи; можно указывать несколько раз",
    )
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
        "--output",
        type=Path,
        default=None,
        help="сохранить сырые результаты в .json или .csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="логировать итерации алгоритмов",
    )
    args = parser.parse_args()

    if args.all_problems and args.problem:
        parser.error(
            "Нельзя одновременно использовать --problem и --all-problems"
        )
    if args.output is not None and args.output.suffix.lower() not in {
        ".json",
        ".csv",
    }:
        parser.error("--output поддерживает только .json и .csv")

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    if args.all_problems:
        problem_names = list_problems()
    elif args.problem:
        problem_names = args.problem
    else:
        problem_names = ["rastrigin"]
    dims = args.dim or [2]

    records = run_experiments(
        problem_names,
        dims,
        args.iterations,
        args.agents,
        args.runs,
        args.seed,
        max_fev=args.max_fev,
        target=args.target,
        budget=args.budget,
        iteration_info=args.verbose,
    )
    rows = summarize_records(records, target=args.target)
    print(format_rows(rows))

    if args.output is not None:
        save_records(records, args.output)
        print(f"Результаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
