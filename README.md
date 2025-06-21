# Metaheuristics​-Lib

Учебная Python-библиотека, реализующая классические метаэвристические алгоритмы для задач глобальной оптимизации. Разрабатывалась как модульная основа для исследовательских и прикладных экспериментов, с упором на поддержку параллельных вычислений (multiprocessing и MPI).

---

## Особенности

- Набор алгоритмов: Artificial Bee Colony, Crow Search Algorithm, Grey Wolf Optimizer, Whale Optimization Algorithm и Emperor Penguin Optimizer.
- Поддержка многопроцессной и распределённой (MPI) обработки популяции.
- Инструменты: логирование, визуализация, сбор метрик (AUC, скорость сходимости, time-to-target и др.).
- Чёткая модульная архитектура: легко расширять за счёт новых задач или алгоритмов.

---

## Структура проекта

```
metaheuristics_lib/
├── metaheuristics_lib/
│   ├── algorithms/
│   │   ├── bee_colony/
│   │   │   ├── artificial_bee_colony.py
│   │   │   ├── artificial_bee_colony_mp.py
│   │   │   └── artificial_bee_colony_mpi.py
│   │   ├── crow_search/
│   │   │   ├── crow_search_algorithm.py
│   │   │   ├── crow_search_algorithm_mp.py
│   │   │   └── crow_search_algorithm_mpi.py
│   │   ├── emperor_pengiun/
│   │   │   ├── emperor_penguin_optimizer.py
│   │   │   ├── emperor_penguin_optimizer_mp.py
│   │   │   └── emperor_penguin_optimizer_mpi.py
│   │   ├── grey_wolf/
│   │   │   ├── grey_wolf_optimizer.py
│   │   │   ├── grey_wolf_optimizer_mp.py
│   │   │   └── grey_wolf_optimizer_mpi.py
│   │   ├── whale/
│   │   │   ├── whale_optimization_algorithm.py
│   │   │   ├── whale_optimization_algorithm_mp.py
│   │   │   └── whale_optimization_algorithm_mpi.py
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

Пространство имён `metaheuristics_lib/` содержит:

- `core/` — абстрактные классы `BaseAlgorithm`, `Runner`;
- `algorithms/` — модули с реализациями метаэвристик;
- `problems/` — тестовые задачи (в том числе benchmark-функции);
- `utils/` — вспомогательные модули (логирование, метрики, парралелизм, визуализация);
- `examples/` — демонстрационные скрипты для запуска на CPU и в MPI-среде.

---

## Интеграции собственной задачи

Необходимо реализовать функцию, соответствующую следующему шаблону:

```python
def your_function(x: np.ndarray) -> float:
    # возвращает значение функции для данного вектора x
    ...
```

---

## Примеры использования

В директории `examples/` представлены скрипты, демонстрирующие работу алгоритмов ( например, `test.py` ) с демонстрацией:

- выбора алгоритма,
- запуска с разными параметрами,
- построения графиков сходимости.

---

## Дальнейшее развитие

- Расширение набора встроенных задач и алгоритмов;
- Поддержка GPU (через CUDA или OpenCL);
- Разработка веб-интерфейса для удалённого запуска и визуализации;
- Портирование библиотеки на другие языки (например, C++ с обёрткой под Python).
  
---

## Установка

```bash
git clone https://github.com/McMorsh/metaheuristics_lib
cd metaheuristics_lib
```
