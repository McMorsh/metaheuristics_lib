# Metaheuristics Lib

Учебная Python-библиотека метаэвристических алгоритмов глобальной
оптимизации. Поддерживает последовательный режим, multiprocessing и
опциональный MPI.

## Возможности

- Пять алгоритмов: ABC, CSA, GWO, WOA и EPO.
- Единый интерфейс `BaseAlgorithm` и запуск через `Runner`.
- Последовательный, процессный и MPI-режимы вычисления функции.
- Метрики сходимости, логирование и визуализация.
- Тесты, Ruff и strict mypy.

## Структура

```text
metaheuristics_lib/
├── algorithms/
│   ├── bee_colony/
│   ├── crow_search/
│   ├── emperor_penguin/
│   ├── grey_wolf/
│   └── whale/
├── core/
├── problems/
├── utils/
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

## Быстрый старт

```python
from algorithms.whale.whale_optimization_algorithm import (
    WhaleOptimizationAlgorithm,
)
from core.runner import Runner
from problems.continuous import sphere

algorithm = WhaleOptimizationAlgorithm(
    problem=sphere,
    dim=10,
    bounds=(-10.0, 10.0),
    agents=20,
    max_iter=200,
    seed=42,
)

with Runner(algorithm, iteration_info=False) as runner:
    result, history, history_x, elapsed = runner.run()

print(result["minimum_value"])
print(result["minimum_x"])
print(len(history))
```

`result` содержит ключи `minimum_x` и `minimum_value`. История включает
начальную точку, поэтому её длина равна `max_iter + 1`.

Библиотека не настраивает logging при импорте. Чтобы видеть сообщения
`Runner`, настройте logging в приложении, например
`logging.basicConfig(level=logging.INFO)`.

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
from algorithms.whale.whale_optimization_algorithm_mp import (
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
picklable. Значение `processes=1` выполняется последовательно.

## MPI

```python
from algorithms.whale.whale_optimization_algorithm_mpi import (
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

Все ранги выполняют одинаковые обновления, а вычисление функции
распределяется через MPI. Результат печатает только rank 0. `mpi4py`
импортируется лениво: без него библиотека работает в последовательном
и процессном режимах.

## Примеры

```bash
python -m examples.run_serial --iterations 20 --agents 10 --runs 2
python -m examples.run_mp --iterations 20 --agents 10 --processes 1 2
mpiexec -n 4 python -m examples.run_mpi --iterations 20 --agents 10
```

Графики сохраняются в `plots/` или в каталог из `--output-dir`.

## Тесты и проверки

```bash
ruff check .
ruff format --check .
mypy .
pytest -q
pytest -q -m integration
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

Передайте функцию в параметр `problem` любого алгоритма.

## Лицензия

MIT. См. `LICENSE`.
