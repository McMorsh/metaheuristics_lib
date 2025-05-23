from algorithms.whale.woa_mpi import WhaleOptimizationAlgorithmMPI
from core.runner import Runner
from problems.continuous import ackley

if __name__ == "__main__":
    algo = WhaleOptimizationAlgorithmMPI(problem=ackley, dim=2, bounds=[(-10, 10)], agents=50, max_iter=200, seed=42)
    runner = Runner(algo)

    best, history = runner.run()
    print("Best:", best)
    print("History:", history)
