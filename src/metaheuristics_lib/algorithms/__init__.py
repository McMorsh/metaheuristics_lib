"""Метаэвристические алгоритмы оптимизации."""

from .bee_colony.artificial_bee_colony import ArtificialBeeColony
from .bee_colony.artificial_bee_colony_mp import ArtificialBeeColonyMP
from .bee_colony.artificial_bee_colony_mpi import ArtificialBeeColonyMPI
from .crow_search.crow_search_algorithm import CrowSearchAlgorithm
from .crow_search.crow_search_algorithm_mp import CrowSearchAlgorithmMP
from .crow_search.crow_search_algorithm_mpi import CrowSearchAlgorithmMPI
from .emperor_penguin.emperor_penguin_optimizer import (
    EmperorPenguinOptimizer,
)
from .emperor_penguin.emperor_penguin_optimizer_mp import (
    EmperorPenguinOptimizerMP,
)
from .emperor_penguin.emperor_penguin_optimizer_mpi import (
    EmperorPenguinOptimizerMPI,
)
from .grey_wolf.grey_wolf_optimizer import GreyWolfOptimizer
from .grey_wolf.grey_wolf_optimizer_mp import GreyWolfOptimizerMP
from .grey_wolf.grey_wolf_optimizer_mpi import GreyWolfOptimizerMPI
from .whale.whale_optimization_algorithm import WhaleOptimizationAlgorithm
from .whale.whale_optimization_algorithm_mp import (
    WhaleOptimizationAlgorithmMP,
)
from .whale.whale_optimization_algorithm_mpi import (
    WhaleOptimizationAlgorithmMPI,
)

__all__ = [
    "ArtificialBeeColony",
    "ArtificialBeeColonyMP",
    "ArtificialBeeColonyMPI",
    "CrowSearchAlgorithm",
    "CrowSearchAlgorithmMP",
    "CrowSearchAlgorithmMPI",
    "EmperorPenguinOptimizer",
    "EmperorPenguinOptimizerMP",
    "EmperorPenguinOptimizerMPI",
    "GreyWolfOptimizer",
    "GreyWolfOptimizerMP",
    "GreyWolfOptimizerMPI",
    "WhaleOptimizationAlgorithm",
    "WhaleOptimizationAlgorithmMP",
    "WhaleOptimizationAlgorithmMPI",
]
