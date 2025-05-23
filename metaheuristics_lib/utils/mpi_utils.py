import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mpi_evaluate_fitness(positions: np.ndarray, func):
    """
    Parallel fitness evaluation via MPI scatter/gather.
    """
    chunks = np.array_split(positions, size) if rank == 0 else None
    local_chunk = comm.scatter(chunks, root=0)
    local_fit = np.array([func(x) for x in local_chunk])
    gathered = comm.gather(local_fit, root=0)
    if rank == 0:
        return np.concatenate(gathered)
    return None
