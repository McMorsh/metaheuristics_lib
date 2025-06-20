# Metaheuristics​-Lib

Учебная Python​-библиотека, реализующая классические алгоритмы метаэвристики для минимизации/максимизации функций. Задумана как учебный каркас и база для будущих продуктов.

---

## 🚀 Особенности

- **Набор алгоритмов**: Artificial Bee Colony, Particle Swarm Optimization, Grey Wolf Optimizer, Whale Optimization и другие.
- **Универсальная архитектура**: базовый класс `BaseAlgorithm` + интерфейс `Runner` для запуска.
- **Удобные утилиты**: функции оценки качества (`metrics`), визуализации (`plot`) и поддержки экспериментов.
- **Поддержка многократных запусков и повторяемости** через установку `seed`.

---

## 📦 Установка

Просто клонируйте репозиторий:

```bash
git clone https://github.com/McMorsh/metaheuristics_lib
cd metaheuristics_lib
```

---

## 🏠 Структура проекта

```
metaheuristics_lib/
├── metaheuristics_lib/
│   ├── algorithms/
│   │   ├── bee_colony/
│   │   │   ├── abc.py
│   │   │   ├── abc_mp.py
│   │   │   └── abc_mpi.py
│   │   ├── crow_search/
│   │   │   ├── csa.py
│   │   │   ├── csa_mp.py
│   │   │   └── csa_mpi.py
│   │   ├── emperor_pengiun/
│   │   │   ├── epo.py
│   │   │   ├── epo_mp.py
│   │   │   └── epo_mpi.py
│   │   ├── grey_wolf/
│   │   │   ├── gwo.py
│   │   │   ├── gwo_mp.py
│   │   │   └── gwo_mpi.py
│   │   ├── whale/
│   │   │   ├── woa.py
│   │   │   ├── woa_mp.py
│   │   │   └── woa_mpi.py
│   ├── utils/
│   │   ├── algorithm_utils.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── mp_utils.py
│   │   ├── mpi_utils.py
│   │   └── plot.py
│   ├── core/
│   │   ├── algorithm.py
│   │   └── runner.py
│   └── problems/
│       └── continuous.py
├── examples/
│   ├── test.py
│   ├── test_mp.py
└── README.md
```

---

## 🧠 Быстрый старт

```python
from algorithms.bee_colony.abc import ArtificialBeeColony
from problems.continuous import rastrigin
from core.runner import Runner

# Создаём и запускаем алгоритм
abc = ArtificialBeeColony(
    problem=rastrigin,
    dim=10,
    bounds=[-5.12, 5.12],
    agents=50,
    max_iter=100,
    seed=42
)

runner = Runner(algorithm=abc)
result = runner.run()
print("Best value:", result[0]['minimum_value'])
print("Best position:", result[0]['minimum_x'])
```

---

## 🔧 Примеры использования

В каталоге `examples/` вы найдёте сценарии ( например, `test.py` ) с демонстрацией:

- выбора алгоритма,
- запуска с разными параметрами,
- построения графиков сходимости.

## ✅ Планы

- Объединить версии алгоритмов ( например, с MPI/многопроцессностью ) в один класс с параметром окончания.
- Поддержать векторизацию критичных участков для ускорения.
- Расширить набор встроенных целевых функций.
- Интеграция поддержки GPU-ускорения (CUDA, OpenCL) для критичных по вычислительным затратам участков.
- Разработка веб-интерфейса для удалённого управления экспериментами и визуализацией результатов.
