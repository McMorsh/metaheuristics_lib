# Changelog

[English](CHANGELOG.md) | [Русский](CHANGELOG.ru.md)

Все заметные изменения проекта. Формат основан на
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версии следуют [семантическому версионированию](https://semver.org/).

## [Unreleased]

### Added

- Основные `README.md` и `CHANGELOG.md` переведены на английский.
  Русские версии находятся в `README.ru.md` и `CHANGELOG.ru.md`.
- Создан публичный каталог `docs/` для документации по алгоритмам.
- `algorithm.nfev_history` и `algorithm.best_fitness_history` хранят
  лучшее значение после каждой оценки.
- Флаг benchmark `--budget` сравнивает алгоритмы при фиксированном
  числе оценок и печатает колонки `f(x)@budget` и `budget`.
- Метрики `best_at_budget` и `evaluations_to_target` для анализа по
  бюджету вычислений.
- `RunRecord` с сырыми результатами и экспорт в JSON или CSV через
  `--output`.
- Несколько значений `--problem` и `--dim`, а также `--all-problems`
  для запуска по нескольким задачам.
- Сводная статистика benchmark: медиана, стандартное отклонение и
  success rate.

### Changed

- Служебные заметки, планы и инструкции по релизу перенесены из
  `docs/` в `notes/`.
- ABC обновляет глобальный лучший результат сразу после улучшающего
  кандидата, поэтому история бюджета точна внутри итерации.
- Таблица benchmark добавляет колонки задачи, размерности, медианы,
  стандартного отклонения и success rate, когда они применимы.

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

## 0.3.0 - 2026-11-05

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
