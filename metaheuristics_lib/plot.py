from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_surface_3d(
        problem,
        best_pos: Optional[np.ndarray] = None,
        resolution: int = 200,
        title: str = None
):
    """
    Построить 3D‑поверхность задачи, если dim=2.

    - problem: любой BaseProblem с dim=2 и bounds shape=(2,2)
    - best_pos: точка (или массив точек) лучших решений, shape=(2,) или (N,2)
    - resolution: число точек по каждой оси
    - title: заголовок графика
    """
    if problem.dim != 2:
        raise ValueError("plot_surface_2d поддерживает только dim=2")

    # создаём сетку из (resolution x resolution) точек
    x = np.linspace(problem.bounds[0, 0], problem.bounds[0, 1], resolution)
    y = np.linspace(problem.bounds[1, 0], problem.bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)

    # готовим массив (resolution², 2) для однократного вызова fitness
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = problem.fitness(pts).reshape(resolution, resolution)

    # рисуем поверхность
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, alpha=0.8)
    ax.set_xlabel('x₁');
    ax.set_ylabel('x₂');
    ax.set_zlabel('fitness')
    if title:
        ax.set_title(title)

    # пометить лучшее решение
    if best_pos is not None:
        best = np.atleast_2d(best_pos)
        z_best = problem.fitness(best)
        ax.scatter(best[:, 0], best[:, 1], z_best, c='r', s=50, marker='x', label='best')
        ax.legend()

    plt.show()


def plot_contour_2d(
        problem,
        best_pos: Optional[np.ndarray] = None,
        resolution: int = 200,
        levels: int = 50,
        title: str = None
):
    """
    2D‑контурный (лейбл‑) график задачи dim=2.
    """
    if problem.dim != 2:
        raise ValueError("plot_contour_2d поддерживает только dim=2")

    x = np.linspace(problem.bounds[0, 0], problem.bounds[0, 1], resolution)
    y = np.linspace(problem.bounds[1, 0], problem.bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = problem.fitness(pts).reshape(resolution, resolution)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, levels=levels)
    plt.colorbar(cp)
    plt.xlabel('x₁');
    plt.ylabel('x₂')
    if title: plt.title(title)

    if best_pos is not None:
        best = np.atleast_2d(best_pos)
        plt.scatter(best[:, 0], best[:, 1], c='r', marker='x', s=50, label='best')
        plt.legend()

    plt.show()
