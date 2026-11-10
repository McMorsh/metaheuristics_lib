# Changelog

[English](CHANGELOG.md) | [Русский](CHANGELOG.ru.md)

All notable changes to this project are documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- English `README.md` and `CHANGELOG.md` as the primary documentation.
  Russian translations are available in `README.ru.md` and
  `CHANGELOG.ru.md`.
- Public `docs/` directory reserved for algorithm documentation.
- `algorithm.nfev_history` and `algorithm.best_fitness_history` track
  the best value after each evaluation.
- Benchmark option `--budget` compares algorithms at a fixed number of
  evaluations and prints `f(x)@budget` and `budget` columns.
- `best_at_budget` and `evaluations_to_target` metrics for budget-based
  analysis.
- `RunRecord` raw results and export to JSON or CSV via `--output`.
- Multiple `--problem` and `--dim` values plus `--all-problems` for
  multi-problem benchmark runs.
- Benchmark summary statistics: median, standard deviation and success
  rate.

### Changed

- Development notes, plans, and release instructions moved from `docs/`
  to `notes/`.
- ABC updates the global best immediately after an improving
  candidate, so the budget history is exact within an iteration.
- The benchmark table adds problem, dimension, median, standard
  deviation and success columns when they are applicable.

## 0.4.0 - 2026-11-06

### Breaking changes

- Removed the deprecated `process` parameter from the `_mp` wrappers.
  Use `processes`.
- Removed the `metaheuristics_lib.utils.algorithm_utils` module. Use
  `normalize_bounds`, `SerialEvaluator`, and the `BaseAlgorithm`
  helpers.
- Removed the deprecated `run_multiple` function. Use `run_repeated`.
- Removed `summarize_runs` and the `utils.mp_utils` and
  `utils.mpi_utils` modules. Use `summarize_run`,
  `ProcessPoolEvaluator`, and `MPIEvaluator`.

### Added

- `max_fev` and `target` for early stopping; the reason is available in
  `algorithm.stop_reason`.
- Fitness finiteness validation: `NaN`, `inf`, and `-inf` raise
  `ValueError`.
- The `metaheuristics_lib.problems.registry` problem registry with
  bounds and known optima.
- `run_repeated` and `summarize_run` for multiple runs.
- Public `metaheuristics_lib.__version__`.
- MPI scenarios for uneven chunks, objective errors, an empty
  population, and comparison of all five wrappers with the serial
  versions.
- Benchmark options `--max-fev`, `--target`, and `--verbose`; the
  benchmark prints the distribution of stop reasons across runs.

### Changed

- `BaseAlgorithm` validates unknown `**params` and raises `TypeError`.
- Removed the `GreyWolfOptimizer` alias from the MPI module. Use
  `GreyWolfOptimizerMPI`.
- EPO evaluates candidates in batches, ABC uses sequential phases, and
  GWO is vectorized.
- CSA restored to its original logic: `expand_rate` is supported again
  and `flight_length` defaults to 1.0.
- ABC restored sequential phases and `1 / (1 + abs(fitness))`
  selection; quality improved, speed decreased.
- WOA restored to its original logic without the exploration branch.
- EPO selects both temperature branches `T`; the swarm radius `R` is
  modeled on `[0, 2]` instead of `[0, 1]`.
- `Runner` accepts `iteration_info` as a `bool` or an interval.
- `area_under_curve` and `run_repeated` document normalization and seed
  restoration.

## 0.3.0 - 2026-11-05

### Breaking changes

- Migrated the project to the `src` layout. Imports `from core...` and
  `from algorithms...` were replaced with `from metaheuristics_lib...`.

### Added

- The `nfev` counter.
- The `examples.benchmark` benchmark.
- Convergence tests and coverage thresholds of 80/70.
- GitHub Actions with `tests`, `package`, and `mpi` jobs.

### Changed

- ABC and EPO moved to batch fitness evaluation.
- GWO no longer uses a double Python loop.
- `ProcessPoolEvaluator` does not create a pool for small batches.
