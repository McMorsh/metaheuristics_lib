# Metaheuristics​-Lib

Учебная Python-библиотека, реализующая классические метаэвристические алгоритмы для задач глобальной оптимизации. Разрабатывалась как модульная основа для исследовательских и прикладных экспериментов, с упором на поддержку параллельных вычислений (multiprocessing и MPI).

---

## Особенности

- Набор алгоритмов: Artificial Bee Colony, Particle Swarm Optimization, Grey Wolf Optimizer, Whale Optimization и Emperor Penguin Optimizer.
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
- Поддержка Island- и Cellular-моделей параллелизма;
- Портирование библиотеки на другие языки (например, C++ с обёрткой под Python).
  
---

## Установка

```bash
git clone https://github.com/McMorsh/metaheuristics_lib
cd metaheuristics_lib
```
