import numpy as np


def  McCormick(x: np.ndarray) -> float:
    positions = np.atleast_2d(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def griewank(x: np.ndarray) -> float:
    """
    Функция Гриуэнка (Griewank) для вектора x.
    """
    x = np.asarray(x)
    sum_term = np.sum(x ** 2) / 4000.0
    idx = np.arange(1, x.size + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(idx)))
    return 1.0 + sum_term - prod_term


def ackley(x: np.ndarray) -> float:
    """
    Функция Эйкли (Ackley) для вектора x.
    Минимизируется при x = 0.
    """
    for i in range(30000):
        np.cos(i)

    x = np.asarray(x)
    n = x.size

    if n == 0:
        raise ValueError("Вектор x не должен быть пустым")

    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)

    return float(term1 + term2 + 20 + np.e)


def sphere(x: np.ndarray) -> float:
    """
    Функция Сферы: сумма квадратов координат.
    Минимизируется при x = 0.
    """
    x = np.asarray(x)
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """
    Функция Розенброка для вектора x размерности >=2.
    Минимизируется при x = [1,...,1].
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x должен быть одномерным вектором размером >= 2")
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """
    Функция Растригина для вектора x.
    Минимизируется при x = 0.
    """
    x = np.asarray(x)
    n = x.size
    A = 10
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def himmelblau(x: np.ndarray) -> float:
    """
    Функция Химмельблау для двумерного x.
    Минимизируется при (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848).
    """
    x = np.asarray(x)
    if x.shape != (2,):
        raise ValueError("x должен быть вектором длины 2")
    x1, x2 = x
    return float((x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2)


def three_hump_camel(x: np.ndarray) -> float:
    """
    Функция трехгорбого верблюда для двумерного x.
    Минимизируется при x = (0,0).
    """
    x = np.asarray(x)
    if x.shape != (2,):
        raise ValueError("x должен быть вектором длины 2")
    x1, x2 = x
    return float(2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2)
