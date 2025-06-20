import numpy as np


def mccormick(arr: np.ndarray) -> float:
    """
    Функция McCormick для вектора x.
    """
    arr = np.asarray(arr)

    if arr.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = np.split(arr, 2)

    return float(np.sum(np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1))


def griewank(x: np.ndarray) -> float:
    """
    Функция Гриуэнка (Griewank) для вектора x.
    """
    x = np.asarray(x)
    n = x.size

    if n == 0:
        raise ValueError("Вектор x не должен быть пустым")

    sum_term = np.sum(x ** 2) / 4000.0
    idx = np.arange(1, n + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(idx)))

    return 1.0 + sum_term - prod_term


def ackley(x: np.ndarray) -> float:
    """
    Функция Эйкли (Ackley) для вектора x.
    """
    # for i in range(100000):
    #      np.cos(i)

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
    Функция Сферы для вектора x.
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Вектор x не должен быть пустым")

    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """
    Функция Розенброка для вектора x.
    """
    x = np.asarray(x)

    if x.size < 2:
        raise ValueError("Вектор x не должен быть меньше 2")

    return float(np.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """
    Функция Растригина для вектора x.
    """
    x = np.asarray(x)
    n = x.size
    A = 10

    if n == 0:
        raise ValueError("Вектор x не должен быть пустым")

    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def himmelblau(arr: np.ndarray) -> float:
    """
    Функция Химмельблау для вектора x.
    """
    arr = np.asarray(arr)

    if arr.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = np.split(arr, 2)

    return float(np.sum((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2))


def three_hump_camel(arr: np.ndarray) -> float:
    """
    Функция трехгорбого верблюда для вектора x.
    """
    arr = np.asarray(arr)

    if arr.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = np.split(arr, 2)

    return float(np.sum(2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2))
