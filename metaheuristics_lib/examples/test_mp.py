from asyncio import sleep

from algorithms.bee_colony.abc_mp import ArtificialBeeColonyMP
from algorithms.crow_search.csa_mp import CrowSearchAlgorithmMP
from algorithms.emperor_penguin.epo_mp import EmperorPenguinOptimizerMP
from algorithms.grey_wolf.gwo_mp import GreyWolfOptimizerMP
from algorithms.whale.woa_mp import WhaleOptimizationAlgorithmMP
from core.runner import Runner
from problems.continuous import sphere
from utils.metrics import best_so_far, area_under_curve, time_to_target


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


if __name__ == '__main__':
    csa()
    sleep(0.5)

    woa()
    sleep(0.5)

    gwo()
    sleep(0.5)

    epo()
    sleep(0.5)

    abc()
