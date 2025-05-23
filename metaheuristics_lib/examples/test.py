from time import sleep

from algorithms.bee_colony.abc import ArtificialBeeColony
from algorithms.crow_search.csa import CrowSearchAlgorithm
from algorithms.emperor_penguin.epo import EmperorPenguinOptimizer
from algorithms.grey_wolf.gwo import GreyWolfOptimizer
from algorithms.whale.woa import WhaleOptimizationAlgorithm
from core.runner import Runner
from problems.continuous import sphere
from utils.metrics import best_so_far, area_under_curve, time_to_target


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
    algo = WhaleOptimizationAlgorithm(sphere, 1, [(-10, 10)], 20, 200, 42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(0.5)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


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
    algo = ArtificialBeeColony(sphere, 10, [(-10, 10)], 20, 200, 42)
    runner = Runner(algo, False)

    best, history = runner.run()
    sleep(1)
    print("Best:", best)
    print("History:", history)
    print("Best so far", best_so_far(history))
    print("area_under_curve:", area_under_curve(history))
    print("time_to_target:", time_to_target(history, target = 0.7))


if __name__ == '__main__':
    # csa()
    # sleep(0.5)

    woa()

    # sleep(0.5)
    #
    # gwo()
    # sleep(0.5)
    #
    # epo()
    # sleep(0.5)
    #
    # abc()
