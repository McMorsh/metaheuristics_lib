# Руководства

[English](README.md) | [Русский](README.ru.md)

Практическое руководство по `metaheuristics_lib`: установка, публичный
API, ранняя остановка, бюджет вычислений, benchmark и параллельные
режимы.

## Содержание

1. [Установка](#установка)
2. [Быстрый старт](#быстрый-старт)
3. [Публичный API](#публичный-api)
4. [Runner](#runner)
5. [Ранняя остановка](#ранняя-остановка)
6. [Бюджет вычислений](#бюджет-вычислений)
7. [Benchmark](#benchmark)
8. [Параллельные режимы](#параллельные-режимы)
9. [Ограничения](#ограничения)
10. [Ссылки](#ссылки)

## Установка

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

Для MPI добавьте extra `mpi` и установите OpenMPI или MPICH:

```bash
pip install -e ".[dev,plot,mpi]"
```

Команды примеров требуют editable-установки, потому что `examples/`
не входит в wheel.

## Быстрый старт

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

`result` содержит `minimum_x` и `minimum_value`. `history` содержит
лучшее значение после инициализации и после каждой итерации.
`history_x` содержит соответствующие позиции. `elapsed` - время
выполнения в секундах.

## Публичный API

Фасад пакета реэкспортирует стабильный API:

- алгоритмы: `ArtificialBeeColony`, `CrowSearchAlgorithm`,
  `EmperorPenguinOptimizer`, `GreyWolfOptimizer`,
  `WhaleOptimizationAlgorithm`;
- ядро: `BaseAlgorithm`, `Runner`, `Evaluator`, `SerialEvaluator`,
  `ProcessPoolEvaluator`, `MPIEvaluator`, `BoundsLike`, `Objective`,
  `OptimizationResult`, `normalize_bounds`;
- задачи: `sphere`, `rastrigin`, `ackley`, `rosenbrock`, `griewank`,
  `himmelblau`, `mccormick`, `three_hump_camel`, `PROBLEMS`,
  `ProblemSpec`, `get_problem`, `list_problems`;
- версия: `__version__`.

Внутренние модули доступны как `metaheuristics_lib.core`,
`metaheuristics_lib.algorithms`, `metaheuristics_lib.problems` и
`metaheuristics_lib.utils`. Отдельных top-level пакетов `core`,
`algorithms`, `problems` и `utils` нет.

## Runner

`Runner` принимает `iteration_info` как `False`, `True` или
положительное целое:

```python
with mhl.Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()
```

- `False` отключает логи итераций;
- `True` логирует каждую итерацию;
- целое число логирует каждую N-ю итерацию.

Runner закрывает ресурсы алгоритма после `run`. Используйте
контекстный менеджер, если запускаете один алгоритм несколько раз,
особенно с пулом процессов.

Библиотека не настраивает logging при импорте. Чтобы видеть сообщения
Runner, настройте logging в приложении:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Ранняя остановка

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

- `target` останавливает алгоритм, когда лучшее значение не больше
  целевого.
- `max_fev` - мягкий лимит `nfev`. Он проверяется на границах
  итераций, поэтому фактическое число вычислений может превысить
  лимит в пределах одной итерации.
- `algorithm.stop_reason` принимает `"target"`, `"max_fev"` или
  `None`. `target` имеет приоритет над `max_fev`.

При ранней остановке история Runner короче `max_iter + 1`.

## Бюджет вычислений

Счётчик `nfev` показывает число обращений к целевой функции за
текущий запуск. Он сбрасывается в `initialize` и учитывает каждую
позицию пакета, включая повторную оценку известной точки.

Для анализа бюджета есть два атрибута:

- `algorithm.nfev_history`: значение `nfev` после каждой оценки;
- `algorithm.best_fitness_history`: лучшее значение в той же точке.

При пакетной оценке одна запись приходится на весь пакет, поэтому
длины историй совпадают.

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

`best_at_budget` возвращает лучшее значение в последней точке истории
с `nfev <= budget` или `None`, если бюджет меньше первой точки.
`evaluations_to_target` возвращает число оценок в первой точке, где
цель достигнута, или `None`.

## Benchmark

### Базовый запуск

```bash
python -m examples.benchmark --iterations 100 --agents 30 --runs 3
```

Задача по умолчанию - `rastrigin`, размерность - `2`, seed - `42`.
Таблица показывает среднее лучшее значение, `nfev`, время и причину
остановки для всех пяти алгоритмов.

### Несколько задач и размерностей

```bash
python -m examples.benchmark \
    --problem sphere --problem ackley \
    --dim 2 --dim 10 \
    --runs 3
```

`--problem` и `--dim` можно указывать несколько раз.
`--all-problems` запускает все задачи из реестра. `--problem` и
`--all-problems` нельзя использовать одновременно.

Задачи с фиксированной размерностью должны использовать именно её.
Например, `himmelblau` определена для `dim=2`.

### Сравнение по бюджету

```bash
python -m examples.benchmark \
    --problem sphere --dim 2 --runs 3 \
    --budget 3000
```

`--budget` сравнивает алгоритмы при фиксированном числе оценок. Если
`--max-fev` не задан, он становится равным бюджету. Лимит остаётся
мягким, поэтому фактический `nfev` может немного превысить бюджет.

Таблица добавляет:

- `f(x)@budget`: среднее лучшее значение на бюджете;
- `budget`: сколько запусков действительно израсходовало бюджет и
  имеет измерение на нём, например `2/3`. Если ни один запуск не
  измерен, выводится `0/N`.

### Ранняя остановка в benchmark

```bash
python -m examples.benchmark \
    --problem rastrigin --dim 10 \
    --max-fev 2000 --target 1e-6 --verbose
```

`--verbose` логирует итерации. Причины остановки агрегируются по
запускам, например `target 2/3, max_fev 1/3`. Прочерк означает, что
все запуски дошли до `max_iter`.

### Сырые результаты

```bash
python -m examples.benchmark \
    --problem sphere --problem ackley \
    --dim 2 --dim 10 --runs 3 \
    --output results.csv
```

`--output` сохраняет сырые строки `RunRecord`. Формат выбирается по
расширению: `.json` или `.csv`.

JSON содержит список объектов. CSV содержит колонки `algorithm`,
`problem`, `dim`, `seed`, `best`, `nfev`, `stop_reason`, `elapsed`,
`budget` и `best_at_budget`.

### Колонки таблицы

Таблица добавляет колонки только когда они применимы:

| Колонка | Когда появляется |
|---|---|
| `Задача`, `dim` | несколько задач или размерностей |
| `медиана`, `std` | больше одного запуска |
| `success` | задан `--target` |
| `f(x)@budget`, `budget` | задан `--budget` |

`std` - выборочное стандартное отклонение. `success` - доля
запусков, достигших цели.

## Параллельные режимы

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

`processes=None` использует число ядер CPU. `mp_context` выбирает
контекст multiprocessing, например `spawn` или `fork`.

На Windows и macOS по умолчанию используется `spawn`. Скрипты должны
защищать точку входа:

```python
if __name__ == "__main__":
    main()
```

Пример команды:

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

Запуск:

```bash
mpiexec -n 4 python -m examples.run_mpi --iterations 100 --agents 20
```

Все ранги выполняют одинаковые обновления, а целевая функция
вычисляется через `MPIEvaluator`. Результат печатает только rank 0.
`mpi4py` импортируется лениво, поэтому библиотека работает без него
в последовательном и процессном режимах.

Если системного MPI нет и нет прав root, на Linux можно поставить
готовый рантайм из PyPI:

```bash
pip install "mpich>=5" "mpi4py>=4"
```

## Ограничения

- `max_fev` мягкий. Он проверяется на границах итераций, поэтому
  запуск может превысить лимит в пределах одной итерации.
- Для пакетных алгоритмов история бюджета содержит одну точку на
  пакет, а не на позицию. ABC оценивает кандидатов по одному, поэтому
  его история точнее внутри итерации.
- В мультизадачном benchmark задачи с фиксированной размерностью
  требуют ровно одну размерность.
- `std` - выборочное стандартное отклонение. Для одного запуска
  `std=0.0`.
- Success rate использует небольшой допуск при сравнении с целью.
- Пакет `examples` не входит в wheel.

## Ссылки

- [README](../README.ru.md)
- [История изменений](../CHANGELOG.ru.md)
- [GitHub Releases](https://github.com/McMorsh/metaheuristics_lib/releases)
- [English version](README.md)
