"""Демонстрация CSA и WOA на функции Растригина."""

import argparse
from pathlib import Path

from metaheuristics_lib import (
    CrowSearchAlgorithm,
    Runner,
    WhaleOptimizationAlgorithm,
    rastrigin,
)
from metaheuristics_lib.utils.metrics import (
    area_under_curve,
    best_so_far,
    run_multiple,
    summarize_runs,
    time_to_target,
)
from metaheuristics_lib.utils.plot import plot_convergence


def test_alg(
    iterations: int,
    agents: int,
    output_dir: Path,
) -> None:
    """Запускает CSA и сохраняет график сходимости."""
    algorithm = CrowSearchAlgorithm(
        rastrigin,
        1,
        (-5.12, 5.12),
        agents,
        iterations,
        42,
        flight_length=1.0,
    )

    with Runner(algorithm) as runner:
        result, history, _, elapsed = runner.run()

    print("Лучшее значение функции:", result["minimum_value"])
    print("Позиция лучшего решения:", result["minimum_x"])
    print("Общее время выполнения:", elapsed, "сек")
    print("Лучшее значение за все итерации:", best_so_far(history))
    print("Площадь под кривой сходимости:", area_under_curve(history))
    print(
        "Время достижения целевого значения:",
        time_to_target(history, target=0.7),
    )

    plot_convergence(
        history,
        labels="CSA",
        title="Сходимость CSA на функции Растригина",
        save_path=output_dir / "csa.png",
    )


def multi_run_test(
    iterations: int,
    agents: int,
    runs: int,
    output_dir: Path,
) -> None:
    """Запускает WOA несколько раз и печатает статистику."""
    algorithm = WhaleOptimizationAlgorithm(
        rastrigin,
        1,
        (-5.12, 5.12),
        agents,
        iterations,
        42,
    )

    with Runner(algorithm, iteration_info=False) as runner:
        results = run_multiple(runner, runs, seed=42)

    print(f"Статистика по {runs} запускам:")
    print(summarize_runs(results, target=0.0))

    plot_convergence(
        results[0]["history_of_best_fitness"],
        labels=[f"Запуск {index + 1}" for index in range(runs)],
        title="Сходимость WOA",
        save_path=output_dir / "woa_runs.png",
    )
    print("Графики сохранены в", output_dir)


def main() -> None:
    """Разбирает аргументы и запускает демонстрации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Тест Crow Search Algorithm")
    test_alg(args.iterations, args.agents, args.output_dir)

    print("\nМногократный тест Whale Optimization Algorithm")
    multi_run_test(
        args.iterations,
        args.agents,
        args.runs,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
