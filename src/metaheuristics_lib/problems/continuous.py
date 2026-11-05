"""Непрерывные тестовые функции для проверки алгоритмов."""

import numpy as np
import numpy.typing as npt


def _as_vector(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Преобразует вход в одномерный вектор float.

    Args:
        values: входные данные.

    Returns:
        Одномерный массив чисел.

    Raises:
        ValueError: если вход не является одномерным.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Вектор x должен быть одномерным")
    return array


def mccormick(values: npt.ArrayLike) -> float:
    """Вычисляет функцию McCormick.

    Args:
        values: вектор из двух координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор не одномерный или имеет размер не 2.
    """
    array = _as_vector(values)
    if array.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = array
    return float(np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0)


def griewank(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Griewank.

    Args:
        values: непустой вектор координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор пуст или не одномерный.
    """
    array = _as_vector(values)
    size = array.size
    if size == 0:
        raise ValueError("Вектор x не должен быть пустым")

    sum_term = float(np.sum(array**2) / 4000.0)
    indices = np.arange(1, size + 1)
    product_term = float(np.prod(np.cos(array / np.sqrt(indices))))
    return 1.0 + sum_term - product_term


def ackley(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Ackley.

    Args:
        values: непустой вектор координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор пуст или не одномерный.
    """
    array = _as_vector(values)
    size = array.size
    if size == 0:
        raise ValueError("Вектор x не должен быть пустым")

    sum_squares = float(np.sum(array**2))
    sum_cosines = float(np.sum(np.cos(2.0 * np.pi * array)))
    first = -20.0 * np.exp(-0.2 * np.sqrt(sum_squares / size))
    second = -np.exp(sum_cosines / size)
    return float(first + second + 20.0 + np.e)


def sphere(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Sphere.

    Args:
        values: непустой вектор координат.

    Returns:
        Сумма квадратов координат.

    Raises:
        ValueError: если вектор пуст или не одномерный.
    """
    array = _as_vector(values)
    if array.size == 0:
        raise ValueError("Вектор x не должен быть пустым")
    return float(np.sum(array**2))


def rosenbrock(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Rosenbrock.

    Args:
        values: вектор минимум из двух координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор короче двух элементов или не одномерный.
    """
    array = _as_vector(values)
    if array.size < 2:
        raise ValueError("Вектор x не должен быть меньше 2")

    return float(
        np.sum(
            (1.0 - array[:-1]) ** 2
            + 100.0 * (array[1:] - array[:-1] ** 2) ** 2
        )
    )


def rastrigin(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Rastrigin.

    Args:
        values: непустой вектор координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор пуст или не одномерный.
    """
    array = _as_vector(values)
    size = array.size
    if size == 0:
        raise ValueError("Вектор x не должен быть пустым")

    amplitude = 10.0
    return float(
        amplitude * size
        + np.sum(array**2 - amplitude * np.cos(2.0 * np.pi * array))
    )


def himmelblau(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Himmelblau.

    Args:
        values: вектор из двух координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор не одномерный или имеет размер не 2.
    """
    array = _as_vector(values)
    if array.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = array
    return float((x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2)


def three_hump_camel(values: npt.ArrayLike) -> float:
    """Вычисляет функцию Three-Hump Camel.

    Args:
        values: вектор из двух координат.

    Returns:
        Значение функции.

    Raises:
        ValueError: если вектор не одномерный или имеет размер не 2.
    """
    array = _as_vector(values)
    if array.size != 2:
        raise ValueError("Вектор x должен быть размерностью 2")

    x, y = array
    return float(2.0 * x**2 - 1.05 * x**4 + x**6 / 6.0 + x * y + y**2)
