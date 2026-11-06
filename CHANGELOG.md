# Changelog

Все заметные изменения проекта. Формат основан на Keep a Changelog,
версии следуют семантическому версионированию.

## 0.4.0 - 2026-11-06

### Breaking changes

- Удалён устаревший параметр `process` в `_mp`-обёртках. Используйте
  `processes`.
- Удалён модуль `metaheuristics_lib.utils.algorithm_utils`.
  Используйте `normalize_bounds`, `SerialEvaluator` и помощники
  `BaseAlgorithm`.
- Удалена устаревшая функция `run_multiple`. Используйте
  `run_repeated`.
- Удалены `summarize_runs` и модули `utils.mp_utils` и
  `utils.mpi_utils`. Используйте `summarize_run`,
  `ProcessPoolEvaluator` и `MPIEvaluator`.

### Added

- `max_fev` и `target` для ранней остановки, причина доступна в
  `algorithm.stop_reason`.
- Проверка конечности fitness: `NaN`, `inf` и `-inf` вызывают
  `ValueError`.
- Реестр задач `metaheuristics_lib.problems.registry` с границами и
  известными оптимумами.
- `run_repeated` и `summarize_run` для нескольких запусков.
- Публичный `metaheuristics_lib.__version__`.
- MPI-сценарии для неравных чанков, ошибки objective, пустой
  популяции и сравнения всех пяти обёрток с serial.
- Benchmark принимает `--max-fev`, `--target` и `--verbose`,
  печатает распределение причин остановки по запускам.

### Changed

- `BaseAlgorithm` проверяет неизвестные `**params` и поднимает
  `TypeError`.
- Удалён алиас `GreyWolfOptimizer` из MPI-модуля. Используйте
  `GreyWolfOptimizerMPI`.
- EPO оценивает кандидатов пакетно, ABC использует
  последовательные фазы, GWO векторизован.
- CSA возвращён к исходной логике: `expand_rate` снова
  поддерживается, `flight_length` по умолчанию 1.0.
- ABC: возвращены последовательные фазы и отбор
  `1 / (1 + abs(fitness))`; качество выросло, скорость снизилась.
- WOA возвращён к исходной логике без исследовательской ветви.
- EPO выбирает обе ветви температуры `T`; радиус стаи `R`
  моделируется на `[0, 2]`, а не на `[0, 1]`.
- `Runner` принимает `iteration_info` как `bool` или интервал.
- `area_under_curve` и `run_repeated` документируют нормировку и
  восстановление seed.

## 0.3.0 - 2026-11-04

### Breaking changes

- Проект переведён на `src`-layout. Импорты `from core...` и
  `from algorithms...` заменены на `from metaheuristics_lib...`.

### Added

- Счётчик `nfev`.
- Benchmark `examples.benchmark`.
- Тесты сходимости и покрытие с порогами 80/70.
- GitHub Actions с jobs `tests`, `package` и `mpi`.

### Changed

- ABC и EPO переведены на пакетную оценку fitness.
- GWO избавлен от двойного Python-цикла.
- `ProcessPoolEvaluator` не создаёт пул для малых пакетов.
