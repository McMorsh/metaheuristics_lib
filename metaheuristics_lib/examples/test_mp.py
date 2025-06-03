from asyncio import sleep

import pandas as pd

from algorithms.bee_colony.abc_mp import ArtificialBeeColonyMP
from algorithms.crow_search.csa_mp import CrowSearchAlgorithmMP
from algorithms.emperor_penguin.epo_mp import EmperorPenguinOptimizerMP
from algorithms.grey_wolf.gwo_mp import GreyWolfOptimizerMP
from algorithms.whale.woa_mp import WhaleOptimizationAlgorithmMP
from core.runner import Runner
from plot import plot_speedup_different_pools
from problems.continuous import *
from utils.metrics import *


def csa():
    algo = CrowSearchAlgorithmMP(sphere, 10, [(-10, 10)], 20, 200, 42, flight_length=1, process=4)
    runner = Runner(algo, False)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def woa():
    algo = WhaleOptimizationAlgorithmMP(sphere, 10, [(-10, 10)], 20, 20, 42, process=2)
    runner = Runner(algo, False)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def gwo():
    algo = GreyWolfOptimizerMP(sphere, 10, [(-10, 10)], 20, 20, seed=42, process=2)
    runner = Runner(algo, False)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def epo():
    algo = EmperorPenguinOptimizerMP(sphere, 10, [(-10, 10)], 20, 20, seed=42, process=2)
    runner = Runner(algo, False)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def abc():
    algo = ArtificialBeeColonyMP(sphere, 10, [(-10, 10)], 20, 50, 42, process=2)
    runner = Runner(algo, False)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def test_all_algorithms():
    functions = [
        # ("Rastrigin", rastrigin, (-5.12, 5.12), 0, 1),
        # ("Three Hump Camel", three_hump_camel, [(-5, 5), (-5, 5)], 0, 2),
        # ("Sphere", sphere, (-100, 100), 0, 10),
        # ("Rosenbrock", rosenbrock, (-100, 100), 0, 10),
        # ("Himmelblau", himmelblau, [(-5, 5), (-5, 5)], 0, 2),
        # ("Griewank", griewank, (-100, 100), 0, 1),
        # ("McCormick", mccormick, [(-1.5, 4), (-3, 4)], -1.9133, 2),
        ("Ackley", ackley, (-5, 5), 0, 30),
    ]

    algorithms = [
        ("Whale Optimization Algorithm", WhaleOptimizationAlgorithmMP),
        # ("Crow Search Algorithm", CrowSearchAlgorithmMP),
        # ("Grey Wolf Optimizer", GreyWolfOptimizerMP),
        # ("Emperor Penguin Optimizer", EmperorPenguinOptimizerMP),
        # ("Artificial Bee Colony", ArtificialBeeColonyMP)
    ]

    pools = [1, 2, 4, 6]  # Процессы

    agents = 20
    max_iterations = 100
    seed = 1

    list_of_results = []

    for p in pools:
        for func_name, func, bounds, target, dim in functions:
            print("-" * 25)
            print(f"Testing on function: {func_name} with Thread: {p}\n")
            for algo_name, AlgoClass in algorithms:
                sleep(0.5)
                algo = AlgoClass(func, dim, bounds, agents, max_iterations, processes=p)

                runner = Runner(algo, True)

                result = run_multiple(runner, 3, seed)

                list_of_results.append({
                    "Function": func_name,
                    "Algorithm": algo_name,
                    "Pools": p,
                    **summarize_runs(result, target)
                })

    save_path = fr"G:\Code\metaheuristics_lib\results_mp\experiment_results.csv"
    df = pd.DataFrame(list_of_results)
    df.to_csv(save_path, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path}\n")


def plot_all():
    df = pd.read_csv(fr"G:\Code\metaheuristics_lib\results_mp\experiment_results.csv", sep=',')

    plot_speedup_different_pools(
        df['time'],
        df['Pools'],
        title=f"График ускорения WOA при выполнении Ackley с искусственным замедлением",
        save_path=fr"G:\Code\metaheuristics_lib\results_mp\WOA_time.png"
    )


if __name__ == '__main__':
    # woa()

    test_all_algorithms()

    plot_all()

    df_loaded = pd.read_csv(fr"G:\Code\metaheuristics_lib\results_mp\experiment_results.csv", sep=',')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df_loaded)
