from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class BaseAlgorithm(ABC):
    """
    Абстрактный базовый класс для всех метаэвристических алгоритмов.
    Определяет единый интерфейс и жизненный цикл алгоритма.
    """

    def __init__(self, problem: Any,
                 dim,
                 bounds,
                 agents: Optional[int] = 100,
                 max_iter: Optional[int] = 50,
                 seed: Optional[int] = None,
                 **params: Any):
        """
        Инициализация оптимизационного алгоритма.

        Args:
            problem (Any): Объект задачи оптимизации. Должен реализовывать методы `fitness(position)` и `random_solution()`.
            dim (int): Размерность пространства решений.
            bounds (Tuple[float, float] или np.ndarray): Границы поиска для каждой переменной (общие или индивидуальные).
            agents (int, optional): Размер популяции агентов (частиц, особей и т.п.). По умолчанию 100.
            max_iter (int, optional): Максимальное количество итераций. По умолчанию 50.
            seed (int, optional): Зерно генератора случайных чисел для воспроизводимости. По умолчанию None.
            **params: Дополнительные параметры алгоритма (например, коэффициенты скорости, инерции и т.д.).
        """
        self.problem = problem
        self.dim = dim
        self.bounds = bounds
        self.agents = agents
        self.max_iter = max_iter

        self.seed = seed

        self.params: Dict[str, Any] = params

        self.history: List[float] = []

        self.best_solution: Any = None
        self.best_fitness: float = float('inf')

    @abstractmethod
    def initialize(self) -> None:
        """
        Подготовительный шаг перед началом итераций.
        Здесь обычно создаются начальная популяция, стая частиц или аналог.
        """
        ...

    @abstractmethod
    def iterate(self) -> None:
        """
        Одна итерация алгоритма.
        Должна обновить внутреннее состояние (популяцию, позиции частиц и т.д.)
        и при необходимости обновить best_solution и best_fitness.
        """
        ...

    @abstractmethod
    def get_result(self) -> Any:
        """
        По окончании run возвращает результат (может быть равно best_solution,
        или содержать дополнительно статистику).
        """
        ...
