# Metaheuristics Lib

[English](README.md) | [Русский](README.ru.md)

An educational Python library of metaheuristic global optimization
algorithms. It supports sequential execution, multiprocessing, and
optional MPI.

## Features

- Five algorithms: ABC, CSA, GWO, WOA, and EPO.
- A common `BaseAlgorithm` interface and the `Runner` driver.
- Sequential, process-based, and MPI evaluation of the objective.
- A shared `Evaluator` abstraction, the `nfev` counter, and
  evaluation-budget histories.
- Convergence metrics, logging, and plotting.
- Benchmark with raw CSV/JSON export, multi-problem runs, and
  summary statistics.
- Tests, coverage, Ruff, and strict mypy.

## Project layout

```text
metaheuristics_lib/
├── src/
│   └── metaheuristics_lib/
│       ├── algorithms/
│       │   ├── bee_colony/
│       │   ├── crow_search/
│       │   ├── emperor_penguin/
│       │   ├── grey_wolf/
│       │   └── whale/
│       ├── core/
│       ├── problems/
│       ├── utils/
│       ├── __init__.py   # public facade
│       └── py.typed
├── examples/
├── tests/
├── docs/                 # reserved for algorithm documentation
├── notes/                # local development notes, not tracked
├── pyproject.toml
├── README.md
├── README.ru.md
├── CHANGELOG.md
├── CHANGELOG.ru.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/McMorsh/metaheuristics_lib
cd metaheuristics_lib
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,plot]"
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,plot]"
```

For MPI, install the `mpi` extra and a system MPI implementation
(OpenMPI or MPICH):

```bash
pip install -e ".[dev,plot,mpi]"
```

## Public API

The `metaheuristics_lib` package re-exports the algorithms, `Runner`,
`Evaluator`, bounds helpers, and benchmark problems. Internal modules
are available as `metaheuristics_lib.core...`,
`metaheuristics_lib.algorithms...`,
`metaheuristics_lib.problems...`, and
`metaheuristics_lib.utils...`.

The old top-level packages `core`, `utils`, `problems`, and
`algorithms` no longer exist. Use the qualified imports shown above.

## Quick start

```python
import metaheuristics_lib as mhl

algorithm = mhl.WhaleOptimizationAlgorithm(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
)

with mhl.Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()

print(result["minimum_value"])
print(result["minimum_x"])
print(len(history))
print(mhl.__version__)
```

`result` contains the keys `minimum_x` and `minimum_value`. The
history includes the initial point. Without early stopping its length
is `max_iter + 1`. If the algorithm stops before the last iteration,
the history is shorter.

`iteration_info` accepts `False`, `True`, or a positive integer.
`False` disables iteration logging, `True` logs every iteration, and
an integer sets the interval.

The `nfev` counter reports the number of objective evaluations for
the current run. It is reset in `initialize` and counts every position
in a batch, including repeated evaluations of a known point. In MPI
the counter grows by the global batch size on every rank, so it is
identical everywhere and does not need to be summed across ranks.

`algorithm.nfev_history` and `algorithm.best_fitness_history` record
the best value after each evaluation. For a batched evaluation one
entry covers the whole batch, so the histories have equal length.

## Early stopping

An algorithm can stop when it reaches a target fitness value or an
evaluation budget:

```python
import metaheuristics_lib as mhl

algorithm = mhl.WhaleOptimizationAlgorithm(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
    target=1e-6,
    max_fev=5000,
)
```

- `target` stops the algorithm when the best value is less than or
  equal to `target`.
- `max_fev` sets a soft limit on `nfev`. It is checked at iteration
  boundaries, so the actual number of evaluations can exceed the limit
  within one iteration.
- `algorithm.stop_reason` is `"target"`, `"max_fev"`, or `None`.

The library does not configure logging on import. To see `Runner`
messages, configure logging in the application, for example with
`logging.basicConfig(level=logging.INFO)`.

## Metrics for multiple runs

```python
import metaheuristics_lib as mhl
from metaheuristics_lib.utils.metrics import run_repeated, summarize_run

algorithm = mhl.WhaleOptimizationAlgorithm(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
)

with mhl.Runner(algorithm, iteration_info=False) as runner:
    result = run_repeated(runner, runs=10, seed=42)

print(summarize_run(result, target=0.0))
```

`run_repeated` returns a single dictionary with the results of all
runs.

For budget-based comparisons use `best_at_budget` and
`evaluations_to_target`:

```python
from metaheuristics_lib.utils.metrics import (
    best_at_budget,
    evaluations_to_target,
)

value = best_at_budget(
    algorithm.nfev_history,
    algorithm.best_fitness_history,
    budget=500,
)
count = evaluations_to_target(
    algorithm.nfev_history,
    algorithm.best_fitness_history,
    target=1e-6,
)
```

`best_at_budget` returns the best value at the last history point
with `nfev <= budget`, or `None` if the budget is below the first
point. `evaluations_to_target` returns the evaluation count at the
first point that reaches the target.

## Search bounds

Supported forms:

- `(low, high)` for all dimensions;
- `[(low, high), ...]` with `dim` entries;
- a `numpy.ndarray` of shape `(2,)` or `(dim, 2)`.

Invalid shapes, `NaN`, infinities, and `low >= high` raise
`ValueError`.

## Parallel execution

The `processes` parameter sets the number of worker processes:

```python
import metaheuristics_lib as mhl
from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithmMP,
)

algorithm = WhaleOptimizationAlgorithmMP(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
    processes=4,
)
```

The objective must be defined at module level and be picklable. A
value of `processes=1` runs sequentially. Batches with fewer than four
positions are also evaluated in the current process because starting a
pool costs more than the computation.

You can select the multiprocessing context explicitly:

```python
import multiprocessing
import sys

import metaheuristics_lib as mhl
from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithmMP,
)

start_method = (
    "fork"
    if sys.platform.startswith("linux")
    and "fork" in multiprocessing.get_all_start_methods()
    else "spawn"
)

algorithm = WhaleOptimizationAlgorithmMP(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    processes=2,
    mp_context=multiprocessing.get_context(start_method),
)
```

This is useful in environments where the default forkserver is
disabled. On macOS and Windows the example uses `spawn`.

## MPI

```python
import metaheuristics_lib as mhl
from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithmMPI,
)

algorithm = WhaleOptimizationAlgorithmMPI(
    problem=mhl.sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
)
```

Run it with:

```bash
mpiexec -n 4 python -m examples.run_mpi --iterations 100 --agents 20
```

If no system MPI is installed and you do not have root access, you can
install a runtime from PyPI on Linux:

```bash
pip install "mpich>=5" "mpi4py>=4"
```

All ranks perform the same updates, while objective evaluation is
distributed through MPI. Only rank 0 prints the result. `mpi4py` is
imported lazily, so the library still works in sequential and
process-based modes without it.

## Examples

```bash
python -m examples.run_serial --iterations 20 --agents 10 --runs 2
python -m examples.run_mp --iterations 20 --agents 10 --processes 1 2
python -m examples.benchmark --iterations 100 --agents 30 --runs 3
python -m examples.benchmark --iterations 100 --agents 30 --runs 3 \
    --budget 3000
python -m examples.benchmark --all-problems --dim 2 --runs 3
python -m examples.benchmark --problem sphere --problem ackley \
    --dim 2 --dim 10 --runs 3 --output results.csv
python -m examples.benchmark --problem rastrigin --dim 10 \
    --max-fev 2000 --target 1e-6 --verbose
mpiexec -n 4 python -m examples.run_mpi --iterations 20 --agents 10
```

Plots are saved to `plots/` or to the directory passed with
`--output-dir`. `examples.benchmark` prints the mean best value,
`nfev`, elapsed time, and the stop reason for all five algorithms.
`--problem` can be repeated, `--all-problems` runs the whole registry,
and `--dim` can be repeated for problems with variable dimension.
`--output` saves raw `RunRecord` rows to JSON or CSV; the format is
chosen by the file extension. For multiple runs the table also shows
the median and standard deviation, and with `--target` it shows the
success rate. `--max-fev` and `--target` enable early stopping, and
`--verbose` logs iterations. `--budget` compares the algorithms at a
fixed number of evaluations and adds `f(x)@budget` and `budget`
columns; `budget` shows how many runs actually reached the budget and
have a measurement at it, for example `2/3`. If no run can be measured
at the budget, `f(x)@budget` is `-` and `budget` is `0/N`. Stop
reasons are aggregated across runs, for example
`target 2/3, max_fev 1/3`; `-` means that all runs reached
`max_iter`. The example commands require an editable install:
`pip install -e ".[dev,plot]"`.

## Problem registry

The standard benchmark problems are collected in
`metaheuristics_lib.problems.registry`:

```python
import metaheuristics_lib as mhl

print(mhl.list_problems())

problem = mhl.get_problem("rastrigin")
print(problem.function)
print(problem.dim)
print(problem.bounds)
print(problem.optimum)
```

For problems with variable dimension, `dim` is `None`. Fixed
two-dimensional problems provide `optimum_x`. The benchmark accepts
registry names through `--problem`.

## Tests and checks

```bash
ruff check .
ruff format --check .
mypy .
pytest -q
pytest -q -m integration
pytest -q \
    --cov=src/metaheuristics_lib/core \
    --cov=src/metaheuristics_lib/utils \
    --cov-fail-under=80
pytest -q \
    --cov=src/metaheuristics_lib/algorithms \
    --cov=src/metaheuristics_lib/problems \
    --cov-fail-under=70
```

The integration tests start a real process pool and may be unavailable
in restricted sandbox environments. Tests that require `mpi4py` skip
automatically when it is not installed.

## Custom problem

```python
import numpy as np
import numpy.typing as npt


def my_function(x: npt.NDArray[np.float64]) -> float:
    """Return the objective value."""
    return float(np.sum(x**2))
```

Pass the function as the `problem` argument of any algorithm. It must
return a finite number: `NaN`, `inf`, and `-inf` raise `ValueError`
with the position that produced the invalid value.

## Documentation

The `docs/` directory is reserved for algorithm documentation.
Development plans and release instructions are kept in `notes/` and
are not tracked. See `CHANGELOG.md` for the release history; the
Russian translation is in `CHANGELOG.ru.md`.

## License

MIT. See `LICENSE`.
