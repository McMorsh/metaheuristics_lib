# Guides

[English](README.md) | [Русский](README.ru.md)

Practical guide to `metaheuristics_lib`: installation, the public API,
early stopping, evaluation budgets, benchmark runs, and parallel
execution.

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Public API](#public-api)
4. [Runner](#runner)
5. [Early stopping](#early-stopping)
6. [Evaluation budget](#evaluation-budget)
7. [Benchmark](#benchmark)
8. [Parallel execution](#parallel-execution)
9. [Limitations](#limitations)
10. [Links](#links)

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

For MPI, add the `mpi` extra and install OpenMPI or MPICH:

```bash
pip install -e ".[dev,plot,mpi]"
```

The example commands require an editable install because `examples/` is
not included in the wheel.

## Quick start

```python
import metaheuristics_lib as mhl

algorithm = mhl.WhaleOptimizationAlgorithm(
    problem=mhl.sphere,
    dim=2,
    bounds=(-5.12, 5.12),
    agents=20,
    max_iter=100,
    seed=42,
)

with mhl.Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()

print(result["minimum_value"])
print(result["minimum_x"])
print(algorithm.nfev)
print(algorithm.stop_reason)
```

`result` contains `minimum_x` and `minimum_value`. `history` contains
the best value after initialization and after every iteration.
`history_x` contains the corresponding positions. `elapsed` is the
elapsed time in seconds.

## Public API

The package facade re-exports the stable API:

- algorithms: `ArtificialBeeColony`, `CrowSearchAlgorithm`,
  `EmperorPenguinOptimizer`, `GreyWolfOptimizer`,
  `WhaleOptimizationAlgorithm`;
- core: `BaseAlgorithm`, `Runner`, `Evaluator`, `SerialEvaluator`,
  `ProcessPoolEvaluator`, `MPIEvaluator`, `BoundsLike`, `Objective`,
  `OptimizationResult`, `normalize_bounds`;
- problems: `sphere`, `rastrigin`, `ackley`, `rosenbrock`, `griewank`,
  `himmelblau`, `mccormick`, `three_hump_camel`, `PROBLEMS`,
  `ProblemSpec`, `get_problem`, `list_problems`;
- version: `__version__`.

Internal modules remain available as `metaheuristics_lib.core`,
`metaheuristics_lib.algorithms`, `metaheuristics_lib.problems`, and
`metaheuristics_lib.utils`. There are no top-level `core`, `algorithms`,
`problems`, or `utils` packages.

## Runner

`Runner` accepts `iteration_info` as `False`, `True`, or a positive
integer:

```python
with mhl.Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()
```

- `False` disables iteration logging;
- `True` logs every iteration;
- an integer logs every N-th iteration.

The runner closes algorithm resources after `run`. Use the context
manager when you run the same algorithm repeatedly, especially with a
process pool.

The library does not configure logging on import. To see runner
messages, configure logging in the application:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Early stopping

```python
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
  equal to the target.
- `max_fev` is a soft limit on `nfev`. It is checked at iteration
  boundaries, so the actual number of evaluations can exceed the limit
  within one iteration.
- `algorithm.stop_reason` is `"target"`, `"max_fev"`, or `None`.
  `target` takes precedence over `max_fev`.

When early stopping is triggered, the runner history is shorter than
`max_iter + 1`.

## Evaluation budget

The `nfev` counter reports the number of objective evaluations for the
current run. It is reset in `initialize` and counts every position in a
batch, including repeated evaluations of a known point.

Two attributes support budget analysis:

- `algorithm.nfev_history`: the `nfev` value after each evaluation;
- `algorithm.best_fitness_history`: the best value at the same point.

For a batched evaluation one entry covers the whole batch, so the
histories have equal length.

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

`best_at_budget` returns the best value at the last history point with
`nfev <= budget`, or `None` if the budget is below the first point.
`evaluations_to_target` returns the evaluation count at the first point
that reaches the target, or `None`.

## Benchmark

### Basic run

```bash
python -m examples.benchmark --iterations 100 --agents 30 --runs 3
```

The default problem is `rastrigin`, the default dimension is `2`, and
the default seed is `42`. The table reports the mean best value,
`nfev`, elapsed time, and the stop reason for all five algorithms.

### Multiple problems and dimensions

```bash
python -m examples.benchmark \
    --problem sphere --problem ackley \
    --dim 2 --dim 10 \
    --runs 3
```

`--problem` and `--dim` can be repeated. `--all-problems` runs every
problem from the registry. `--problem` and `--all-problems` are
mutually exclusive.

Problems with a fixed dimension must use exactly that dimension. For
example, `himmelblau` is defined for `dim=2`.

### Budget comparison

```bash
python -m examples.benchmark \
    --problem sphere --dim 2 --runs 3 \
    --budget 3000
```

`--budget` compares algorithms at a fixed number of evaluations. If
`--max-fev` is not set, it becomes equal to the budget. The limit stays
soft, so actual `nfev` can slightly exceed the budget.

The table adds:

- `f(x)@budget`: mean best value at the budget;
- `budget`: how many runs reached the budget and have a measurement at
  it, for example `2/3`. If no run can be measured, it is `0/N`.

### Early stopping in benchmark

```bash
python -m examples.benchmark \
    --problem rastrigin --dim 10 \
    --max-fev 2000 --target 1e-6 --verbose
```

`--verbose` logs iterations. Stop reasons are aggregated across runs,
for example `target 2/3, max_fev 1/3`. A dash means that all runs
reached `max_iter`.

### Raw results

```bash
python -m examples.benchmark \
    --problem sphere --problem ackley \
    --dim 2 --dim 10 --runs 3 \
    --output results.csv
```

`--output` saves raw `RunRecord` rows. The format is selected by the
extension: `.json` or `.csv`.

JSON contains a list of objects. CSV contains the columns
`algorithm`, `problem`, `dim`, `seed`, `best`, `nfev`, `stop_reason`,
`elapsed`, `budget`, and `best_at_budget`.

### Table columns

The table adds columns only when they are applicable:

| Column | When it appears |
|---|---|
| `Problem`, `dim` | multiple problems or dimensions |
| `median`, `std` | more than one run |
| `success` | `--target` is set |
| `f(x)@budget`, `budget` | `--budget` is set |

`std` is the sample standard deviation. `success` is the fraction of
runs that reached the target.

## Parallel execution

### Multiprocessing

```python
from metaheuristics_lib import Runner, sphere
from metaheuristics_lib.algorithms import WhaleOptimizationAlgorithmMP

algorithm = WhaleOptimizationAlgorithmMP(
    sphere,
    10,
    (-10.0, 10.0),
    agents=20,
    max_iter=100,
    seed=42,
    processes=4,
)

with Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()
```

`processes=None` uses the number of CPU cores. `mp_context` selects a
multiprocessing context, for example `spawn` or `fork`.

On Windows and macOS the default start method is `spawn`. Scripts must
guard their entry point:

```python
if __name__ == "__main__":
    main()
```

Example command:

```bash
python -m examples.run_mp --iterations 20 --agents 10 --processes 1 2 4
```

### MPI

```python
from metaheuristics_lib import Runner, sphere
from metaheuristics_lib.algorithms import WhaleOptimizationAlgorithmMPI

algorithm = WhaleOptimizationAlgorithmMPI(
    sphere,
    10,
    (-10.0, 10.0),
    agents=20,
    max_iter=100,
    seed=42,
)

with Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()

if algorithm.is_root:
    print(result["minimum_value"])
```

Run it with:

```bash
mpiexec -n 4 python -m examples.run_mpi --iterations 100 --agents 20
```

All ranks perform the same updates, while the objective is evaluated
through `MPIEvaluator`. Only rank 0 prints the result. `mpi4py` is
imported lazily, so the library works without it in sequential and
process-based modes.

If no system MPI is installed and you do not have root access, you can
install a runtime from PyPI on Linux:

```bash
pip install "mpich>=5" "mpi4py>=4"
```

## Limitations

- `max_fev` is soft. It is checked at iteration boundaries, so a run
  can exceed the limit within one iteration.
- For batched algorithms the budget history has one point per batch,
  not per position. ABC evaluates candidates one by one, so its history
  is more precise within an iteration.
- Multi-problem benchmark runs require a single dimension for problems
  with a fixed dimension.
- `std` is the sample standard deviation. A single run has `std=0.0`.
- Success rate uses a small tolerance when comparing with the target.
- The `examples` package is not included in the wheel.

## Links

- [README](../README.md)
- [Changelog](../CHANGELOG.md)
- [GitHub Releases](https://github.com/McMorsh/metaheuristics_lib/releases)
- [Russian version](README.ru.md)
