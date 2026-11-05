"""Демонстрация WOA с пулом процессов и графиком ускорения."""

import argparse
import multiprocessing
import sys
from multiprocessing.context import BaseContext
from pathlib import Path

from metaheuristics_lib import Runner, sphere
from metaheuristics_lib.algorithms import WhaleOptimizationAlgorithmMP
from metaheuristics_lib.utils.plot import plot_speedup_different_pools


def _process_context() -> BaseContext | None:
    """Возвращает fork-контекст, если он доступен."""
    if (
        sys.platform.startswith("linux")
        and "fork" in multiprocessing.get_all_start_methods()
    ):
        return multiprocessing.get_context("fork")
    return None


def test_alg(
    iterations: int,
    agents: int,
    process_counts: list[int],
    output_dir: Path,
) -> None:
    """Запускает WOA с разным числом процессов."""
    elapsed_times: list[float] = []
    process_context = _process_context()

    for processes in process_counts:
        algorithm = WhaleOptimizationAlgorithmMP(
            sphere,
            10,
            (-10.0, 10.0),
            agents,
            iterations,
            42,
            processes=processes,
            mp_context=process_context,
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
