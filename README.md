# Metaheuristics Lib

Учебная Python-библиотека метаэвристических алгоритмов глобальной
оптимизации. Поддерживает последовательный режим, multiprocessing и
опциональный MPI.

## Возможности

- Пять алгоритмов: ABC, CSA, GWO, WOA и EPO.
- Единый интерфейс `BaseAlgorithm` и запуск через `Runner`.
- Последовательный, процессный и MPI-режимы вычисления функции.
- Единый `Evaluator` и счётчик обращений `nfev`.
- Метрики сходимости, логирование и визуализация.
- Тесты, покрытие, Ruff и strict mypy.

## Структура

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
│       ├── __init__.py   # публичный фасад
│       └── py.typed
├── examples/
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

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

## Публичный API

Пакет `metaheuristics_lib` реэкспортирует алгоритмы, `Runner`,
`Evaluator`, границы и тестовые функции. Внутренние модули доступны
как `metaheuristics_lib.core...`, `metaheuristics_lib.algorithms...`,
`metaheuristics_lib.problems...` и `metaheuristics_lib.utils...`.
Отдельных top-level пакетов `core`, `utils`, `problems` и `algorithms`
больше нет. Пример использования приведен ниже.

## Быстрый старт

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

`result` содержит ключи `minimum_x` и `minimum_value`. История включает
начальную точку. Без ранней остановки её длина равна `max_iter + 1`,
при остановке она короче.

`iteration_info` принимает `False`, `True` или положительное целое.
`False` отключает лог итераций, `True` логирует каждую итерацию,
целое число задаёт интервал.

Счётчик `algorithm.nfev` показывает число обращений к целевой функции
за текущий запуск. Он сбрасывается в `initialize` и учитывает каждую
позицию пакета, включая повторную оценку уже известной точки. В MPI
счётчик увеличивается на размер глобального пакета на каждом ранге,
поэтому он одинаков везде и не требует суммирования по рангам.

## Ранняя остановка

Алгоритм можно остановить по достижению целевого значения fitness
или по лимиту вычислений:

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
  `target`.
- `max_fev` задаёт мягкий лимит `nfev`. Он проверяется на границах
  итераций, поэтому фактическое число вычислений может превысить
  лимит в пределах одной итерации.
- `algorithm.stop_reason` принимает значение `"target"`, `"max_fev"`
  или `None`.

Библиотека не настраивает logging при импорте. Чтобы видеть сообщения
`Runner`, настройте logging в приложении, например
`logging.basicConfig(level=logging.INFO)`.

## Метрики нескольких запусков

```python
import metaheuristics_lib as mhl
from metaheuristics_lib.utils.metrics import run_repeated, summarize_run

with mhl.Runner(algorithm, iteration_info=False) as runner:
    result = run_repeated(runner, runs=10, seed=42)

print(summarize_run(result, target=0.0))
```

`run_repeated` возвращает один словарь с результатами всех запусков.

## Границы поиска

Поддерживаются:

- `(low, high)` для всех размерностей;
- `[(low, high), ...]` длиной `dim`;
- `numpy.ndarray` формы `(2,)` или `(dim, 2)`.

Неверные формы, `NaN`, бесконечности и `low >= high` приводят к
`ValueError`.

## Параллельный режим

Параметр `processes` задаёт число процессов:

```python
from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithmMP,
)

algorithm = WhaleOptimizationAlgorithmMP(
    problem=sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
    processes=4,
)
```

Целевая функция должна быть определена на верхнем уровне модуля и быть
picklable. Значение `processes=1` выполняется последовательно. Пакеты
меньше четырёх позиций тоже считаются в текущем процессе: запуск пула
для них дороже вычисления.

При необходимости можно явно выбрать контекст multiprocessing:

```python
import multiprocessing
import sys

start_method = (
    "fork"
    if sys.platform.startswith("linux")
    and "fork" in multiprocessing.get_all_start_methods()
    else "spawn"
)

algorithm = WhaleOptimizationAlgorithmMP(
    problem=sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    processes=2,
    mp_context=multiprocessing.get_context(start_method),
)
```

Это полезно в окружениях, где стандартный forkserver запрещен.
На macOS и Windows пример использует `spawn`.

## MPI

```python
from metaheuristics_lib.algorithms import (
    WhaleOptimizationAlgorithmMPI,
)

algorithm = WhaleOptimizationAlgorithmMPI(
    problem=sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
)
```

Запуск:

```bash
mpiexec -n 4 python -m examples.run_mpi --iterations 100 --agents 20
```

Если системного MPI нет и нет прав root, на Linux можно поставить
готовый рантайм из PyPI:

```bash
pip install "mpich>=5" "mpi4py>=4"
```

Все ранги выполняют одинаковые обновления, а вычисление функции
распределяется через MPI. Результат печатает только rank 0. `mpi4py`
импортируется лениво: без него библиотека работает в последовательном
и процессном режимах.

## Примеры

```bash
python -m examples.run_serial --iterations 20 --agents 10 --runs 2
python -m examples.run_mp --iterations 20 --agents 10 --processes 1 2
python -m examples.benchmark --iterations 100 --agents 30 --runs 3
python -m examples.benchmark --problem rastrigin --dim 10 \
    --max-fev 2000 --target 1e-6 --verbose
mpiexec -n 4 python -m examples.run_mpi --iterations 20 --agents 10
```

Графики сохраняются в `plots/` или в каталог из `--output-dir`.
`examples.benchmark` печатает среднее лучшее значение, `nfev`, время
и причину остановки для всех пяти алгоритмов на задаче из реестра.
`--max-fev` и `--target` включают раннюю остановку, а `--verbose`
логирует итерации. Причина остановки агрегируется по запускам,
например `target 2/3, max_fev 1/3`; `-` означает, что все запуски
дошли до `max_iter`. Команды примеров требуют editable-установки
пакета: `pip install -e ".[dev,plot]"`.

## Реестр задач

Стандартные функции собраны в `metaheuristics_lib.problems.registry`:

```python
import metaheuristics_lib as mhl

print(mhl.list_problems())

problem = mhl.get_problem("rastrigin")
print(problem.function)
print(problem.dim)
print(problem.bounds)
print(problem.optimum)
```

Для задач с переменной размерностью `dim` равен `None`. Для
фиксированных двумерных задач задано `optimum_x`. Benchmark принимает
имена задач из реестра через `--problem`.

## Тесты и проверки

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

Интеграционные тесты запускают реальный пул процессов и могут быть
недоступны в ограниченных sandbox-окружениях.

## Своя задача

```python
import numpy as np


def my_function(x: np.ndarray) -> float:
    """Возвращает значение целевой функции."""
    return float(np.sum(x**2))
```

Передайте функцию в параметр `problem` любого алгоритма. Она должна
возвращать конечное число: `NaN`, `inf` и `-inf` приводят к
`ValueError` с указанием позиции.

## Лицензия

MIT. См. `LICENSE`.
