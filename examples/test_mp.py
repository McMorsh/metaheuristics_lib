"""Демонстрация WOA с пулом процессов и графиком ускорения."""

import argparse
from pathlib import Path

from algorithms.whale.whale_optimization_algorithm_mp import (
    WhaleOptimizationAlgorithmMP,
)
from core.runner import Runner
from problems.continuous import sphere
from utils.plot import plot_speedup_different_pools


def test_alg(
    iterations: int,
    agents: int,
    process_counts: list[int],
    output_dir: Path,
) -> None:
    """Запускает WOA с разным числом процессов."""
    elapsed_times: list[float] = []

    for processes in process_counts:
        algorithm = WhaleOptimizationAlgorithmMP(
            sphere,
            10,
            (-10.0, 10.0),
            agents,
            iterations,
            42,
            processes=processes,
        )

        with Runner(algorithm, iteration_info=False) as runner:
            _, _, _, elapsed = runner.run()

        elapsed_times.append(elapsed)
        print(f"Процессов: {processes}, время: {elapsed:.4f} сек")

    plot_speedup_different_pools(
        elapsed_times,
        process_counts,
        title="Ускорение WOA",
        save_path=output_dir / "woa_speedup.png",
    )


def main() -> None:
    """Разбирает аргументы и запускает демонстрацию."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument(
        "--processes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Тест Whale Optimization Algorithm MP")
    test_alg(
        args.iterations,
        args.agents,
        args.processes,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
