from algorithms.whale.whale_optimization_algorithm_mp import WhaleOptimizationAlgorithmMP
from core.runner import Runner
from problems.continuous import sphere
from utils.plot import plot_speedup_different_pools


def test_alg():
    """
    Тестирование многопроцессорной версии алгоритма Whale Optimization Algorithm (WOA)
    на сферической функции с использованием 2,4,6 процесса и построение графика ускорения
    """
    list_of_process = [2, 4, 6]
    list_of_time = []
    
    for p in list_of_process:
        # 1. Инициализация алгоритма с параметрами:
        # - sphere: тестовая функция
        # - 10: размерность задачи
        # - (-10, 10): границы поиска
        # - 20: количество агентов
        # - 200: максимальное число итераций
        # - 42: seed для воспроизводимости
        # - process=p: количество процессов
        algo = WhaleOptimizationAlgorithmMP(sphere, 10, (-10, 10), 20, 200, 42, process=p)

        # 2. Создание и запуск (Runner) - обертка для выполнения алгоритма
        runner = Runner(algo, False)  # Создаем Runner с отключенным подробным выводом
        _, _, _, total_elapsed = runner.run()

        # 3. Добавление времени выполнения
        list_of_time.append(total_elapsed)
        
    # 4. Построение графика ускорения
    plot_speedup_different_pools(
        list_of_time,
        list_of_process,
        title=f"График ускорения WOA",
        save_path=fr"G:\...\WOA_time.png"
    )
