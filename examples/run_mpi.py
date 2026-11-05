"""Демонстрация MPI-версии Whale Optimization Algorithm."""

import argparse

from metaheuristics_lib import Runner, sphere
from metaheuristics_lib.algorithms import WhaleOptimizationAlgorithmMPI


def main() -> None:
    """Запускает MPI-версию WOA и печатает результат на rank 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--agents", type=int, default=20)
    args = parser.parse_args()

    algorithm = WhaleOptimizationAlgorithmMPI(
        sphere,
        10,
        (-10.0, 10.0),
        agents=args.agents,
        max_iter=args.iterations,
        seed=42,
    )

    with Runner(algorithm, iteration_info=False) as runner:
        result, _, _, elapsed = runner.run()

    if algorithm.is_root:
        print("Лучшее значение функции:", result["minimum_value"])
        print("Позиция лучшего решения:", result["minimum_x"])
        print("Время выполнения:", elapsed, "сек")


if __name__ == "__main__":
    main()
