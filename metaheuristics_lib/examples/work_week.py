import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, simpson
from scipy.optimize import *
import random, math, time, warnings

from algorithms.bee_colony.abc_mp import ArtificialBeeColonyMP
from algorithms.crow_search.csa_mp import CrowSearchAlgorithmMP
from algorithms.emperor_penguin.epo_mp import EmperorPenguinOptimizerMP
from algorithms.grey_wolf.gwo_mp import GreyWolfOptimizerMP
from algorithms.whale.woa_mp import WhaleOptimizationAlgorithmMP

from core.runner import Runner
from plot import plot_execution_time_comparison, plot_convergence, plot_speedup_different_pools
from utils.metrics import summarize_runs, run_multiple

def func(rho, y):
    '''
    y1=y, y2=y1'=y', y2'=y1''=y''
    y[0]=R(r), y[1]=R'(r)
    '''
    y1 = y[0]
    y2 = y[1]
    lmbd = 2*nu/(1-2*nu)
    t1 = rho ** 2
    t2 = t1 ** 2
    t4 = 2 * m
    t5 = l + t4
    t7 = y2 ** 2
    t8 = t7 ** 2
    t12 = omega ** 2
    t13 = t12 * t1
    t14 = t13 + 1
    t22 = y1 ** 2
    t24 = k ** 2
    t25 = l * t24
    t26 = 3 * l
    t28 = (t25 - t26 + lmbd - t4 + 2) * t1
    t34 = t14 ** 2
    t35 = t34 * l
    t37 = t24 - 3
    t38 = l * t37
    t39 = n / 2
    t42 = t1 * (t38 + m - t39 + lmbd) * t12
    t43 = -m + t39
    t44 = t24 * t43
    t52 = t14 * l
    t54 = t22 ** 2
    t65 = t37 ** 2
    t66 = l * t65
    t67 = 2 * lmbd
    t69 = t24 * (t4 - n + t67)
    t70 = 6 * lmbd
    t72 = (t66 + t69 + n - t70 - 4) * t2
    t89 = t24 ** 2
    return np.vstack([y2,0.1e1 / (5 * t8 * t5 * t2 + 6 * t7 * (t22 * t52 + t28) * t1 + t54 * t35 + t22 * t1 * (2 * t42 + 2 * t38 + 2 * t44 + t4 - n + t67) + t72) / t1 * (-t8 * y2 * t5 * t2 * rho - 3 * t8 * t14 * t2 * y1 * l - 2 * t7 * y2 * (t22 * (t13 - 1) * l + t28) * t1 * rho - 2 * t7 * y1 * (t22 * t35 + t1 * (t42 + t38 + t44 + m - t39 + lmbd)) * t1 - y2 * (t54 * (t13 - 3) * t52 + 2 * t22 * (-l * t37 - t24 * t43 - lmbd - m + t39 + t42) * t1 + t72) * rho + y1 * (t54 * t5 * t34 * t14 + 2 * t22 * t1 * (t1 * (2 * m * t24 + lmbd + t38 - t4 + 2) * t12 + t25 - t26 - t4 + lmbd + 2) * t14 + (-4 + t1 * (-4 + t66 + 2 * m * t89 + t24 * (-4 * m + t67 + 4) + n - t70) * t12 + t66 + t69 + n - t70) * t2))])

def bc(ya,yb):
    '''
    ya[0]=R(rho_0), ya[1]=R'(rho_0)
    yb[0]=R(1),     yb[1]=R'(1)
    '''
    ya1 = ya[0]
    ya2 = ya[1]
    yb1 = yb[0]
    yb2 = yb[1]
    lmbd = 2*nu/(1-2*nu)
    t1 = rho_0 ** 2
    t2 = t1 ** 2
    t3 = 2 * m
    t4 = l + t3
    t6 = ya2 ** 2
    t7 = t6 ** 2
    t9 = omega ** 2
    t11 = t1 * t9 + 1
    t13 = ya1 ** 2
    t15 = k ** 2
    t16 = l * t15
    t24 = t11 ** 2
    t26 = t13 ** 2
    t28 = t15 - 3
    t29 = l * t28
    t30 = n / 2
    t32 = (t29 + m - t30 + lmbd) * t9
    t35 = 2 * t29
    t38 = 2 * t15 * (-m + t30)
    t39 = 2 * lmbd
    t43 = t28 ** 2
    t44 = l * t43
    t46 = t15 * (t3 - n + t39)
    t47 = 6 * lmbd
    t55 = yb2 ** 2
    t56 = t55 ** 2
    t58 = t9 + 1
    t60 = yb1 ** 2
    t68 = t58 ** 2
    t70 = t60 ** 2
    bc1 = 1 / t2 * ya2 * (t7 * t4 * t2 + 2 * t6 * (t13 * t11 * l + (t16 - 3 * l + lmbd - t3 + 2) * t1) * t1 + t26 * t24 * l + t13 * t1 * (2 * t1 * t32 - n + t3 + t35 + t38 + t39) + (t44 + t46 + n - t47 - 4) * t2) / 4
    bc2 = yb2 * (-4 + t56 * t4 + t55 * (2 * l * t58 * t60 - 6 * l - 4 * m + 2 * t16 + t39 + 4) + t70 * t68 * l + t60 * (2 * t32 + t35 + t38 + t3 - n + t39) + t44 + t46 + n - t47) / 4
    return np.array([bc1, bc2]).reshape(-1)

def integrand_Q(rho, y1, y2):
    lmbd = 2*nu/(1-2*nu)
    t1 = rho ** 2
    t5 = omega ** 2
    t6 = t5 ** 2
    t7 = 2 * m
    t10 = y1 ** 2
    t11 = t10 ** 2
    t13 = k ** 2
    t14 = y2 ** 2
    t15 = t13 + t14 - 3
    t16 = l * t15
    t20 = 2 * lmbd
    t21 = 4 * m
    t25 = t15 ** 2
    t27 = t13 ** 2
    t36 = t1 ** 2
    t41 = n / 2
    return (t36 * (t11 * (l + t7) * t6 + t10 * t5 * (4 * m * t13 + 2 * t16 + t20 - t21 + 4) + l * t25 + 2 * m * t27 + t13 * (-t21 + t20 + 4) + t14 * (t7 - n + t20) - 6 * lmbd + n - 4) + 2 * t1 * t10 * (t10 * (l + m) * t5 + t16 + t14 * (-m + t41) + lmbd + m - t41) + t11 * l) * k * math.pi / t1 / rho / 2

def integrand_M(rho, y1, y2):
    lmbd = 2*nu/(1-2*nu)
    t1 = rho ** 2
    t4 = omega ** 2
    t5 = t4 ** 2
    t7 = l + 2 * m
    t9 = y1 ** 2
    t10 = t9 ** 2
    t12 = k ** 2
    t13 = y2 ** 2
    t14 = t12 + t13 - 3
    t16 = 2 * l * t14
    t17 = m * t12
    t19 = 2 * lmbd
    t20 = 4 * m
    t24 = t14 ** 2
    t26 = t12 ** 2
    t37 = t1 ** 2
    return omega * math.pi * t9 * (t37 * (t10 * t7 * t5 + t9 * t4 * (t16 + 4 * t17 + t19 - t20 + 4) + l * t24 + 2 * m * t26 + t12 * (-t20 + t19 + 4) + 2 * m * t13 + t13 * (t19 - n) - 6 * lmbd + n - 4) + t1 * t9 * (2 * t4 * t7 * t9 + t16 + 2 * t17 + t19 - t20 + 4) + t7 * t10) / t1 / rho / 2

def Q(om, x):
    global k, omega
    k = x
    omega = om
    y_appr = [rho.copy(), np.ones(rho.size)]
    res_a = solve_bvp(func, bc, rho, y_appr, tol=1e-6)
    if not res_a.success:
        return np.nan
    return simpson(integrand_Q(res_a.x, res_a.y[0], res_a.y[1]), x=res_a.x)

def M(om, x):
    global k, omega
    k = x
    omega = om
    y_appr = [rho.copy(), np.ones(rho.size)]
    res_a = solve_bvp(func, bc, rho, y_appr, tol=1e-6)
    if not res_a.success:
        return np.nan
    return simpson(integrand_M(res_a.x, res_a.y[0], res_a.y[1]), x=res_a.x, dx=0.001)

def Func_torsion(omegas, nuu, ll, mm, nn):
    global nu, l, m, n
    nu, l, m, n = nuu, ll, mm, nn

    k_val = []
    for om in omegas:
        try:
            root = newton(lambda x: Q(om, x), 1.0, maxiter=50)
        except (RuntimeError, ValueError):
            try:
                root = newton(lambda x: Q(om, x), 1.01, maxiter=50)
            except (RuntimeError, ValueError):
                try:
                    root = brentq(lambda x: Q(om, x), 0.5, 2.0, maxiter=50)
                except (RuntimeError, ValueError):
                    return None, None
        k_val.append(root)

    list3 = [M(omegas[i], k_val[i]) for i in range(len(k_val))]
    return np.array(list3), np.array(k_val)

# Func_torsion - omegas, nu, l0, m0, n0
def test_fun(x: np.ndarray) -> float:
    try:
        if len(x) != 3:
            return 1e6
        _, y1 = Func_torsion(omegas_ref, nu_ref, x[0], x[1], x[2])
        if y1 is None:
            return 1e6
        return float(np.sum((y1 - k_val_ref) ** 2))
    except Exception as e:
        return 1e6

    except Exception as e:
        print(f"[test_fun] Исключение: {e}")
        return 1e6

def test_all_algorithms():
    functions = [
        ("Work_week", test_fun, [(-5, -1), (-15, -10), (-35, -25)], 0, 3)
    ]

    algorithms = [
        ("Whale Optimization Algorithm", WhaleOptimizationAlgorithmMP),
        ("Crow Search Algorithm", CrowSearchAlgorithmMP),
        #("Grey Wolf Optimizer", GreyWolfOptimizerMP),
        ("Emperor Penguin Optimizer", EmperorPenguinOptimizerMP),
        ("Artificial Bee Colony", ArtificialBeeColonyMP)
    ]

    pools = [2, 4]  # Процессы

    agents = 20
    max_iterations = 3
    seed = 1

    list_of_results = []
    list_of_data = []

    for p in pools:
        for func_name, func, bounds, target, dim in functions:
            print("-" * 25)
            print(f"Testing on function: {func_name} with Thread: {p}\n")
            for algo_name, AlgoClass in algorithms:

                time.sleep(0.5)

                if algo_name == "Crow Search Algorithm":
                    algo = AlgoClass(func, dim, bounds, agents, max_iterations, flight_length=1, awareness_prob=0.1,
                                     expand_rate=1.2, processes = p)
                elif algo_name == "Artificial Bee Colony":
                    algo = AlgoClass(func, dim, bounds, agents, max_iterations, limit=25, processes = p)
                else:
                    algo = AlgoClass(func, dim, bounds, agents, max_iterations, processes = p)

                runner = Runner(algo, True)

                result = run_multiple(runner, 1, seed)

                list_of_results.append({
                    "Function": func_name,
                    "Algorithm": algo_name,
                    "Pools": p,
                    **summarize_runs(result, target)
                })

                list_of_data.append({
                    "Function": func_name,
                    "Algorithm": algo_name,
                    "Pools": p,
                    **result[0],
                })


    save_path1 = fr"G:\Code\metaheuristics_lib\week_work\convergence_data.csv"
    df1 = pd.DataFrame(list_of_data)
    df1.to_csv(save_path1, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path1}\n")

    save_path2 = fr"G:\Code\metaheuristics_lib\week_work\experiment_results.csv"
    df2 = pd.DataFrame(list_of_results)
    df2.to_csv(save_path2, index=False, sep=',', encoding='utf-8')
    print(f"\n✅ Results saved to: {save_path2}\n")


def plot_all():
    df_conv = pd.read_csv(fr"G:\Code\metaheuristics_lib\week_work\convergence_data.csv", sep=',')

    # Преобразуем строку истории в массив
    df_conv["history"] = df_conv["history"].apply(eval)  # безопасно, если файл твой
    df_conv["time"] = df_conv["time"].apply(eval)

    # Получаем список уникальных процессов
    pools = df_conv["Pools"].unique()

    for p in pools:
        pool_df = df_conv[df_conv["Pools"] == p]

        histories = []
        labels = []
        times = []

        for _, row in pool_df.iterrows():
            label = row["Algorithm"]
            all_runs = np.array(row["history"])  # shape = (30, итерации)
            all_times = np.array(row["time"])  # shape = (30,)

            mean_history = np.mean(all_runs, axis=0)
            histories.append(mean_history)
            labels.append(label)

            # усредняем время выполнения
            mean_time = np.mean(all_times)  # или np.median(all_times)
            times.append(mean_time)

        # Строим график
        plot_convergence(
            histories,
            labels,
            title=f"Сходимость на функции: test при {p} процессах",
            save_path=fr"G:\Code\metaheuristics_lib\week_work\{p}_conv.png"
        )

        plot_execution_time_comparison(
            times,
            labels,
            title=f"Сравнение времени выполнения алгоритмов на функции: test при {p} процессах",
            save_path=fr"G:\Code\metaheuristics_lib\week_work\{p}_time.png"
        )


    df = pd.read_csv(fr"G:\Code\metaheuristics_lib\week_work\experiment_results.csv", sep=',')
    # Получаем список уникальных алгоритмов
    algs = df["Algorithm"].unique()

    for alg in algs:
        alg_df = df[df["Algorithm"] == alg]

        times_df = []
        pools_df = []

        for _, row in alg_df.iterrows():
            pools_df.append(row["Pools"])
            times_df.append(row["time"])

        plot_speedup_different_pools(
            times_df,
            pools_df,
            title=f"График ускорения {alg} при выполнении Ackley с искусственным замедлением",
            save_path=fr"G:\Code\metaheuristics_lib\results_mp\WOA_time.png"
        )



# Параметры материала (медь)
rho_0 = 0.9
N = 30
rho = np.linspace(rho_0, 1, N)  # радиус цилиндра
nu_ref, l0_ref, m0_ref, n0_ref = 0.346, -2.264, -13.082, -33.375
omegas_ref = np.linspace(0.0, 0.07, 30)  # диапазон частот для меди

# Вычисление эталонных значений (один раз при загрузке модуля)
_, k_val_ref = Func_torsion(omegas_ref, nu_ref, l0_ref, m0_ref, n0_ref)
if k_val_ref is None:
    raise RuntimeError("Не удалось вычислить эталонные значения")

if __name__ == '__main__':

    #test_all_algorithms()

    plot_all()

    df_loaded = pd.read_csv(fr"G:\Code\metaheuristics_lib\week_work\experiment_results.csv", sep=',')

    df = pd.read_csv(fr"G:\Code\metaheuristics_lib\week_work\convergence_data.csv", sep=',')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df_loaded,'\n', df)