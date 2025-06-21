from algorithms.crow_search.crow_search_algorithm import CrowSearchAlgorithm
from algorithms.whale.whale_optimization_algorithm import WhaleOptimizationAlgorithm
from core.runner import Runner
from problems.continuous import rastrigin
from utils.metrics import time_to_target, area_under_curve, best_so_far, run_multiple, summarize_runs
from utils.plot import plot_convergence

def test_alg():
    """
    Тестирование одного алгоритма на одной функции с визуализацией результатов
    """
    # Инициализация алгоритма:
    # - rastrigin: тестовая функция
    # - 1: размерность задачи
    # - (-5.12, 5.12): границы поиска
    # - 20: количество агентов (частиц/особей)
    # - 200: максимальное число итераций
    # - 42: seed для воспроизводимости
    # - flight_length=1: специфический параметр алгоритма CSA
    algo = CrowSearchAlgorithm(rastrigin, 1, (-5.12, 5.12), 20, 200, 42, flight_length=1)

    # Создание и запуск обертки для выполнения алгоритма
    runner = Runner(algo)
    result, history_of_best_fitness, history_of_best_x, total_elapsed = runner.run()
    
    # Вывод результатов
    print("Лучшее значение функции:", result['minimum_value'])
    print("Позиция лучшего решения:", result['minimum_x'])
    print("Общее время выполнения:", total_elapsed, "сек")
    
    # Расчет и вывод дополнительных метрик
    print("Лучшее значение за все итерации:", best_so_far(history_of_best_fitness))
    print("Площадь под кривой сходимости:", area_under_curve(history_of_best_fitness))
    print("Время достижения целевого значения:", time_to_target(history_of_best_fitness, target=0.7))

    # Визуализация сходимости
    plot_convergence(
        history_of_best_fitness,
        labels="CSA",
        title="Сходимость алгоритма CSA на функции Растригина",
        save_path=fr"G:\...\CSA.png"
    )

def multi_run_test():
    """
    Многократное тестирование алгоритма для статистической оценки производительности
    """
    # Инициализация алгоритма (в данном случае Whale Optimization Algorithm)
    # Параметры аналогичны предыдущей функции
    algo = WhaleOptimizationAlgorithm(rastrigin, 1, (-5.12, 5.12), 20, 200)

    runner = Runner(algo)

    result = run_multiple(runner, 50, seed=42)

    # Анализ и вывод суммарной статистики
    # - target=0: целевое значение функции (известный оптимум)
    print("Статистика по 50 запускам:")
    print(summarize_runs(result, target=0))

if __name__ == '__main__':
    print("Тест Crow Search Algorithm")
    test_alg()
    
    print("\nМногократный тест Whale Optimization Algorithm")
    multi_run_test()
