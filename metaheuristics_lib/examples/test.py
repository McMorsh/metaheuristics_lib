from time import sleep
import pandas as pd

from algorithms.bee_colony.abc import ArtificialBeeColony
from algorithms.crow_search.csa import CrowSearchAlgorithm
from algorithms.emperor_penguin.epo import EmperorPenguinOptimizer
from algorithms.grey_wolf.gwo import GreyWolfOptimizer
from algorithms.whale.woa import WhaleOptimizationAlgorithm
from core.runner import Runner
from plot import plot_fitness_function, plot_convergence
from problems.continuous import *
from utils.metrics import best_so_far, area_under_curve, time_to_target, run_multiple, summarize_runs


def csa():
    algo = CrowSearchAlgorithm(sphere, 10, [(-10, 10)], 20, 200, 42, flight_length=1)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(0.5)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


def woa():
    algo = WhaleOptimizationAlgorithm(sphere, 2, [(-10, 10)], 20, 200)
    runner = Runner(algo, False)

    best, history, time = runner.run()
    sleep(0.5)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))

    #plot_fitness_function(sphere, [(-10, 10)], best_pos=best_so_far(history))
    plot_convergence(history)


def gwo():
    algo = GreyWolfOptimizer(sphere, 10, [(-10, 10)], 20, 200)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(0.5)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


def epo():
    algo = EmperorPenguinOptimizer(sphere, 10, [(-10, 10)], 20, 200, seed=42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(1)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


def abc():
    algo = ArtificialBeeColony(mccormick, 10, [(-10, 10)], 20, 200, 42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(1)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


def test_all_algorithms():
    functions = [
        ("Rastrigin", rastrigin, [(-5.12, 5.12)], 0),
        ("Three Hump Camel", three_hump_camel, [(-5,5),(-5,5)], 0),
        ("Sphere", sphere, [(-100, 100)], 0),
        ("Rosenbrock", rosenbrock, [(-100,100)], 0),
        ("Himmelblau", himmelblau, [(-5,5),(-5,5)], 0),
        ("Griewank", griewank, [(-100, 100)], 0),
        ("McCormick", mccormick, [(-1.5, 4), (-3, 4)], 1.9133),
        ("Ackley", ackley, [(-5.12, 5.12)], 0),
    ]

    algorithms = [
        ("Whale Optimization Algorithm", WhaleOptimizationAlgorithm),
        ("Crow Search Algorithm", CrowSearchAlgorithm),
        ("Grey Wolf Optimizer", GreyWolfOptimizer),
        ("Emperor Penguin Optimizer", EmperorPenguinOptimizer),
        ("Artificial Bee Colony", ArtificialBeeColony)
    ]

    dim = 10
    agents = 20
    max_iterations = 100
    seed = 42
    list_of_results = []

    for func_name, func, bounds, target in functions:
        print("-"*25)
        print(f"Testing on function: {func_name}\n")
        for algo_name, AlgoClass in algorithms:

            sleep(0.5)

            algo = AlgoClass(func, dim, bounds, agents, max_iterations)
            runner = Runner(algo, False)

            result = run_multiple(runner, seed)

            list_of_results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                **summarize_runs(result, target)
            })

    save_path = fr"G:\Code\metaheuristics_lib\results\experiment_results.csv"
    df = pd.DataFrame(list_of_results)
    df.to_csv(save_path, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path}\n")

    df_loaded = pd.read_csv(fr"G:\Code\metaheuristics_lib\results\experiment_results.csv", sep=',')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df_loaded)


if __name__ == '__main__':
    #woa()
    test_all_algorithms()