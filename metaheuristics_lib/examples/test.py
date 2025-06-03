from time import sleep

import numpy as np
import pandas as pd

from algorithms.bee_colony.abc import ArtificialBeeColony
from algorithms.crow_search.csa import CrowSearchAlgorithm
from algorithms.emperor_penguin.epo import EmperorPenguinOptimizer
from algorithms.grey_wolf.gwo import GreyWolfOptimizer
from algorithms.whale.woa import WhaleOptimizationAlgorithm
from core.runner import Runner
from plot import *
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
    print("time_to_target:", time_to_target(history, target=0.7))


def woa():
    algo = WhaleOptimizationAlgorithm(sphere, 2, [(-10, 10)], 20, 200)
    runner = Runner(algo, False)

    best, history, time = runner.run()
    sleep(0.5)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))

    # plot_fitness_function(sphere, [(-10, 10)], best_pos=best_so_far(history))
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
    print("time_to_target:", time_to_target(history, target=0.7))


def epo():
    algo = EmperorPenguinOptimizer(sphere, 10, [(-10, 10)], 20, 200, seed=42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(1)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def abc():
    algo = ArtificialBeeColony(mccormick, 10, [(-10, 10)], 20, 200, 42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(1)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target=0.7))


def test_all_algorithms():
    functions = [
        ("Rastrigin", rastrigin, (-5.12, 5.12), 0, 1),
        ("Three Hump Camel", three_hump_camel, [(-5, 5), (-5, 5)], 0, 2),
        ("Sphere", sphere, (-100, 100), 0, 10),
        ("Rosenbrock", rosenbrock, (-100, 100), 0, 10),
        ("Himmelblau", himmelblau, [(-5, 5), (-5, 5)], 0, 2),
        ("Griewank", griewank, (-100, 100), 0, 1),
        ("McCormick", mccormick, [(-1.5, 4), (-3, 4)], -1.9133, 2),
        ("Ackley", ackley, (-5, 5), 0, 30),
    ]

    algorithms = [
        ("Whale Optimization Algorithm", WhaleOptimizationAlgorithm),
        ("Crow Search Algorithm", CrowSearchAlgorithm),
        ("Grey Wolf Optimizer", GreyWolfOptimizer),
        ("Emperor Penguin Optimizer", EmperorPenguinOptimizer),
        ("Artificial Bee Colony", ArtificialBeeColony)
    ]

    agents = 20
    max_iterations = 100
    seed = 1

    list_of_results = []
    list_of_data = []

    for func_name, func, bounds, target, dim in functions:
        print("-" * 25)
        print(f"Testing on function: {func_name}\n")
        for algo_name, AlgoClass in algorithms:

            sleep(0.5)
            if algo_name == "Crow Search Algorithm":
                algo = AlgoClass(func, dim, bounds, agents, max_iterations, flight_length=1, awareness_prob=0.1,
                                 expand_rate=1.2)
            elif algo_name == "Artificial Bee Colony":
                algo = AlgoClass(func, dim, bounds, agents, max_iterations, limit=25)
            else:
                algo = AlgoClass(func, dim, bounds, agents, max_iterations)

            runner = Runner(algo, False)

            result = run_multiple(runner, 50, seed)

            list_of_results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                **summarize_runs(result, target = 0)
            })

            list_of_data.append({
                "Function": func_name,
                "Algorithm": algo_name,
                **result[0],
            })

    save_path1 = fr"G:\Code\metaheuristics_lib\results\convergence_data.csv"
    df1 = pd.DataFrame(list_of_data)
    df1.to_csv(save_path1, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path1}\n")

    save_path2 = fr"G:\Code\metaheuristics_lib\results\experiment_results.csv"
    df2 = pd.DataFrame(list_of_results)
    df2.to_csv(save_path2, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path2}\n")


def plot_all():
    df = pd.read_csv(fr"G:\Code\metaheuristics_lib\results\convergence_data_new.csv", sep=',')

    # Преобразуем строку истории в массив
    df["history"] = df["history"].apply(eval)  # безопасно, если файл твой
    df["time"] = df["time"].apply(eval)

    # Получаем список уникальных функций
    functions = df["Function"].unique()

    for func in functions:
        func_df = df[df["Function"] == func]

        histories = []
        labels = []
        times = []

        for _, row in func_df.iterrows():
            label = row["Algorithm"]
            all_runs = np.array(row["history"])  # shape = (30, итерации)
            all_times = np.array(row["time"])  # shape = (30,)

            mean_history = np.mean(all_runs, axis=0)
            histories.append(mean_history)
            labels.append(label)

            # усредняем время выполнения
            mean_time = np.mean(all_times)  # или np.median(all_times)
            times.append(mean_time)

        # Строим график
        plot_convergence(
            histories,
            labels,
            title=f"Сходимость на функции: {func}",
            save_path = fr"G:\Code\metaheuristics_lib\results\сходимость\{func}_conv.png"
        )
        plot_execution_time_comparison(
            times,
            labels,
            title=f"Сравнение времени выполнения алгоритмов на функции: {func}",
            save_path = fr"G:\Code\metaheuristics_lib\results\время\{func}_time.png"
        )


if __name__ == '__main__':
    # woa()

    #test_all_algorithms()

    plot_all()

    df_loaded = pd.read_csv(fr"G:\Code\metaheuristics_lib\results\experiment_results_new.csv", sep=',')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df_loaded)
