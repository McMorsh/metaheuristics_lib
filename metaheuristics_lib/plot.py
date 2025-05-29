import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_builder(fitness_func, bounds, resolution, title):
    if title is None:
        title = f'Функция {fitness_func.__name__}'

    bounds = np.array(bounds)
    x_min, x_max = bounds[0]

    if bounds.shape[0] > 1:
        y_min, y_max = bounds[1]
    else:
        y_min, y_max = bounds[0]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_func(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z, title

def plot_fitness_function_above(fitness_func, bounds, resolution=100, title=None, best_pos=None, save_path=None):
    """
    Строит контурный график (вид сверху) целевой функции с подписями уровней.

    :param fitness_func: Целевая функция (принимает np.ndarray и возвращает float).
    :param bounds: Список кортежей [(x_min, x_max), (y_min, y_max)] или одного кортежа для обоих.
    :param resolution: Разрешение сетки (кол-во точек по оси).
    :param title: Заголовок графика.
    :param best_pos: Координаты лучшего решения (опционально).
    :param save_path: Путь для сохранения графика (опционально).
    """
    X, Y, Z, title = plot_builder(fitness_func, bounds, resolution, title)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

    if best_pos is not None:
        best_pos = np.atleast_1d(best_pos)
        if best_pos.shape[0] == 2:
            plt.plot(best_pos[0], best_pos[1], 'r.', markersize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_fitness_function(fitness_func, bounds, resolution=100, title = None, best_pos = None, save_path=None):
    """
    Строит 3D-график функции приспособленности для функций.

    :param: fitness_func (callable): Целевая функция, которую необходимо минимизировать.
    :param: bounds (list of tuples): Список кортежей (low, high) с границами для каждой размерности.
    :param: resolution (int): Количество точек по каждой оси.
    :param: title (str): Заголовок графика.
    :param: save_path (str, optional): Путь для сохранения графика. Если None, график отображается на экране.
    """
    X, Y, Z, title = plot_builder(fitness_func, bounds, resolution, title)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness Value')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if best_pos is not None:
        best_pos = np.atleast_1d(best_pos)
        if best_pos.shape[0] == 2:
            z_val = fitness_func(best_pos)
            ax.scatter(best_pos[0], best_pos[1], z_val,
                       color='r', s=100, label='Best Solution', marker='x')
            ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
               
def plot_convergence(history, title='График сходимости', save_path=None):
    """
    Строит график сходимости на основе истории значений целевой функции.

    :param: history (list or np.ndarray): Список значений целевой функции на каждой итерации.
    :param: title (str): Заголовок графика.
    :param: save_path (str, optional): Путь для сохранения графика. Если None, график отображается на экране.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history, label='Лучшее значение')
    plt.xlabel('Итерация')
    plt.ylabel('Значение целевой функции')
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_speedup_different_algo(histories, labels, title='Сравнение скорости сходимости', save_path=None):
    """
    Строит графики сходимости для нескольких алгоритмов.

    :param: histories (list of list of float): Списки значений функции приспособленности на каждой итерации для разных алгоритмов.
    :param: labels (list of str): Названия алгоритмов.
    :param: title (str): Заголовок графика.
    :param: save_path (str, optional): Путь для сохранения графика. Если None, график отображается на экране.
    """

    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    plt.title(title)
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_speedup_different_pools(times, labels=None, title='График ускорения', save_path=None):
    """
    Строит график ускорения на основе времени выполнения при различном количестве потоков.

    :param times (list or np.ndarray): Время выполнения для каждого количества потоков.
    :param labels (list, optional): Метки для оси X (например, количество потоков).
    :param title (str): Заголовок графика.
    :param save_path (str, optional): Путь для сохранения графика. Если None, график отображается на экране.
    """
    times = np.array(times)
    baseline = times[0]
    speedup = baseline / times
    x = labels if labels else range(1, len(times) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, speedup, marker='o', label='Ускорение')
    plt.xlabel('Количество потоков')
    plt.ylabel('Ускорение')
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()