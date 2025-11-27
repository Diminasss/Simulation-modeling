#!/usr/bin/env python3
"""
simulate_streams.py

Симуляция пуассоновского и эрланговского потоков (renewal processes),
сравнение эмпирических и теоретических характеристик, построение графиков
и сохранение результатов.

Зависимости:
- numpy
- pandas
- matplotlib
- (опционально) scipy для статистических тестов

Пример запуска:
python simulate_streams.py --lambda 1.0 --k 3 --T 1000 --window 1.0 --seed 42
"""

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Опционально: использовать scipy для KS-теста, если установлен
try:
    from scipy import stats  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def simulate_renewal(process_type: str, lambda_rate: float, k: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """
    Сымитировать моменты прибытия renewal-процесса на отрезке [0, T].

    process_type: 'poisson' или 'erlang'
    lambda_rate: интенсивность (для экспоненциального mean IA = 1/lambda)
    k: порядок Эрланга (целое >0), игнорируется для 'poisson'
    T: время моделирования
    rng: экземпляр numpy.random.Generator
    """
    t = 0.0
    arrivals: List[float] = []
    if process_type == 'poisson':
        scale = 1.0 / lambda_rate
        while t < T:
            ia = rng.exponential(scale=scale)
            t += ia
            if t <= T:
                arrivals.append(t)
    elif process_type == 'erlang':
        # Параметры так, чтобы mean interarrival = 1/lambda_rate
        # Для эрланга mean = k * scale => scale = 1/(k * lambda_rate)
        scale = 1.0 / (k * lambda_rate)
        shape = k
        while t < T:
            ia = rng.gamma(shape=shape, scale=scale)
            t += ia
            if t <= T:
                arrivals.append(t)
    else:
        raise ValueError("Unknown process_type: use 'poisson' or 'erlang'")
    return np.array(arrivals)


def counts_in_windows(arrivals: np.ndarray, T: float, window: float) -> np.ndarray:
    bins = np.arange(0.0, T + window, window)
    counts, _ = np.histogram(arrivals, bins=bins)
    return counts


def erlang_pdf(x: np.ndarray, k: int, lambda_rate: float) -> np.ndarray:
    # Для эрланга используем theta = k * lambda_rate, scale = 1/theta
    theta = k * lambda_rate
    # pdf = theta^k * x^{k-1} * exp(-theta x) / (k-1)!
    coef = (theta ** k) / math.factorial(k - 1)
    return coef * (x ** (k - 1)) * np.exp(-theta * x)


def save_summary_csv(df: pd.DataFrame, out_dir: str, name: str = "summary.csv") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    return path


def plot_and_save(ia_poisson: np.ndarray, ia_erlang: np.ndarray,
                  counts_poisson: np.ndarray, counts_erlang: np.ndarray,
                  lambda_rate: float, k: int, T: float, window: float,
                  out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Гистограммы интервалов + теоретические плотности
    max_x = max(np.max(ia_poisson) if ia_poisson.size else 0.0,
                np.max(ia_erlang) if ia_erlang.size else 0.0)
    x_vals = np.linspace(0, max(1.0, max_x) * 0.95, 400)

    pdf_exp = lambda_rate * np.exp(-lambda_rate * x_vals)
    pdf_erlang = erlang_pdf(x_vals, k, lambda_rate)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(ia_poisson, bins=60, density=True, alpha=0.7)
    plt.plot(x_vals, pdf_exp, label='theoretical exp pdf')
    plt.title('Интервалы: пуассоновский поток (экспоненциальные IA)')
    plt.xlabel('Интервал')
    plt.ylabel('Плотность')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(ia_erlang, bins=60, density=True, alpha=0.7)
    plt.plot(x_vals, pdf_erlang, label=f'theoretical Erlang(k={k}) pdf')
    plt.title('Интервалы: эрланговский поток')
    plt.xlabel('Интервал')
    plt.ylabel('Плотность')
    plt.legend()

    path1 = os.path.join(out_dir, 'intervals_histograms.png')
    plt.tight_layout()
    plt.savefig(path1, dpi=150)
    plt.close()

    # 2) Гистограммы числа событий в окнах и сравнение с пуассоновским PMF
    max_count = int(max(np.max(counts_poisson) if counts_poisson.size else 0,
                        np.max(counts_erlang) if counts_erlang.size else 0))
    count_vals = np.arange(0, max_count + 1)
    mu_window = lambda_rate * window
    pmf_poisson = [math.exp(-mu_window) * (mu_window ** n) / math.factorial(n) for n in count_vals]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(counts_poisson, bins=np.arange(-0.5, max_count + 1.5, 1), density=True, alpha=0.7)
    plt.plot(count_vals, pmf_poisson, marker='o', linestyle='-', label='Poisson PMF (теор.)')
    plt.title('Распределение числа событий в окнах — Пуассон')
    plt.xlabel('Число событий в окне')
    plt.ylabel('Относительная частота')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(counts_erlang, bins=np.arange(-0.5, max_count + 1.5, 1), density=True, alpha=0.7)
    plt.plot(count_vals, pmf_poisson, marker='o', linestyle='-', label='Poisson PMF (для сравнения)')
    plt.title('Распределение числа событий в окнах — Эрланг')
    plt.xlabel('Число событий в окне')
    plt.ylabel('Относительная частота')
    plt.legend()

    path2 = os.path.join(out_dir, 'counts_histograms.png')
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()

    # 3) Timeline (часть интервала для наглядности)
    # Рисуем только события до min(200, T)
    cutoff = min(200.0, T)
    plt.figure(figsize=(10, 3))
    plt.eventplot([ia_to_timeline(ia_poisson, cutoff), ia_to_timeline(ia_erlang, cutoff)], linelengths=0.8)
    plt.yticks([0, 1], ['Poisson', f'Erlang(k={k})'])
    plt.xlabel('Время')
    plt.title(f'Таймлайны событий (до t={cutoff})')
    path3 = os.path.join(out_dir, 'timelines.png')
    plt.tight_layout()
    plt.savefig(path3, dpi=150)
    plt.close()

    return path1, path2, path3


def ia_to_timeline(arrivals: np.ndarray, cutoff: float) -> np.ndarray:
    if arrivals.size == 0:
        return np.array([])
    return arrivals[arrivals <= cutoff]


def compute_summary(arr_poisson: np.ndarray, arr_erlang: np.ndarray, counts_poisson: np.ndarray, counts_erlang: np.ndarray,
                    lambda_rate: float, k: int, window: float) -> pd.DataFrame:
    ia_poisson = np.diff(np.concatenate(([0.0], arr_poisson))) if arr_poisson.size > 0 else np.array([])
    ia_erlang = np.diff(np.concatenate(([0.0], arr_erlang))) if arr_erlang.size > 0 else np.array([])

    mean_exp_theory = 1.0 / lambda_rate
    var_exp_theory = 1.0 / (lambda_rate ** 2)
    cv_exp_theory = 1.0

    theta = k * lambda_rate
    mean_erlang_theory = k / theta  # = 1/lambda_rate
    var_erlang_theory = k / (theta ** 2)  # = 1/(k * lambda^2)
    cv_erlang_theory = math.sqrt(var_erlang_theory) / mean_erlang_theory

    def safe_mean(x): return float(np.mean(x)) if x.size else float('nan')
    def safe_var(x): return float(np.var(x, ddof=1)) if x.size and x.size > 1 else float('nan')
    def safe_cv(x): return float(np.std(x, ddof=1) / np.mean(x)) if x.size and np.mean(x) != 0 else float('nan')

    summary = pd.DataFrame([
        {
            'process': 'Poisson (exp IA)',
            'arrivals_count': int(arr_poisson.size),
            'ia_mean_empirical': safe_mean(ia_poisson),
            'ia_var_empirical': safe_var(ia_poisson),
            'ia_cv_empirical': safe_cv(ia_poisson),
            'ia_mean_theory': mean_exp_theory,
            'ia_var_theory': var_exp_theory,
            'ia_cv_theory': cv_exp_theory,
            'counts_mean_empirical': float(np.mean(counts_poisson)) if counts_poisson.size else float('nan'),
            'counts_var_empirical': float(np.var(counts_poisson, ddof=1)) if counts_poisson.size and counts_poisson.size > 1 else float('nan'),
            'counts_mean_theory': lambda_rate * window,
            'counts_var_theory_poisson': lambda_rate * window
        },
        {
            'process': f'Erlang (k={k})',
            'arrivals_count': int(arr_erlang.size),
            'ia_mean_empirical': safe_mean(ia_erlang),
            'ia_var_empirical': safe_var(ia_erlang),
            'ia_cv_empirical': safe_cv(ia_erlang),
            'ia_mean_theory': mean_erlang_theory,
            'ia_var_theory': var_erlang_theory,
            'ia_cv_theory': cv_erlang_theory,
            'counts_mean_empirical': float(np.mean(counts_erlang)) if counts_erlang.size else float('nan'),
            'counts_var_empirical': float(np.var(counts_erlang, ddof=1)) if counts_erlang.size and counts_erlang.size > 1 else float('nan'),
            'counts_mean_theory': lambda_rate * window,
            'counts_var_theory_poisson': float('nan')
        }
    ])
    return summary


def optional_stat_tests(ia_poisson: np.ndarray, ia_erlang: np.ndarray):
    results = {}
    if SCIPY_AVAILABLE:
        # KS: сравнить эмпирическое распределение интервалов с теоретической экспонентой
        if ia_poisson.size > 1:
            # kstest requires a callable cdf or distribution name + args. Для экспоненциального из scipy: 'expon' с scale=1/lambda
            # Но проще: применим одновыборочный KS к нормализованным данным
            pass
        # two-sample KS между интервалами
        if ia_poisson.size > 0 and ia_erlang.size > 0:
            ks = stats.ks_2samp(ia_poisson, ia_erlang)
            results['ks_ia_poisson_vs_erlang'] = ks._asdict()
    else:
        results['scipy'] = 'not available'
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Симуляция пуассоновского и эрланговского потоков")
    p.add_argument("--lambda", dest="lambda_rate", type=float, default=1.0, help="интенсивность lambda (по умолчанию 1.0)")
    p.add_argument("--k", type=int, default=3, help="параметр k для распределения Эрланга (целое)")
    p.add_argument("--T", type=float, default=1000.0, help="время моделирования")
    p.add_argument("--window", type=float, default=1.0, help="ширина окна для подсчёта событий")
    p.add_argument("--seed", type=int, default=None, help="seed для генератора случайных чисел (по умолчанию случайный)")
    p.add_argument("--out", type=str, default="results", help="папка для сохранения результатов")
    return p.parse_args()


def main():
    args = parse_args()
    lambda_rate = args.lambda_rate
    k = args.k
    T = args.T
    window = args.window
    seed = args.seed
    out_dir = args.out

    rng = np.random.default_rng(seed)

    print(f"Запуск симуляции: lambda={lambda_rate}, k={k}, T={T}, window={window}, seed={seed}")

    arr_poisson = simulate_renewal('poisson', lambda_rate=lambda_rate, k=k, T=T, rng=rng)
    arr_erlang = simulate_renewal('erlang', lambda_rate=lambda_rate, k=k, T=T, rng=rng)

    ia_poisson = np.diff(np.concatenate(([0.0], arr_poisson))) if arr_poisson.size else np.array([])
    ia_erlang = np.diff(np.concatenate(([0.0], arr_erlang))) if arr_erlang.size else np.array([])

    counts_poisson = counts_in_windows(arr_poisson, T, window)
    counts_erlang = counts_in_windows(arr_erlang, T, window)

    summary = compute_summary(arr_poisson, arr_erlang, counts_poisson, counts_erlang, lambda_rate, k, window)
    print("\nСводные статистики:")
    print(summary.to_string(index=False))

    csv_path = save_summary_csv(summary, out_dir)
    print(f"\nСводная таблица сохранена в: {csv_path}")

    # Стат. тесты (опционально)
    tests = optional_stat_tests(ia_poisson, ia_erlang)
    if tests:
        print("\nРезультаты опциональных статистических тестов:")
        print(tests)

    # Построить и сохранить графики
    p1, p2, p3 = plot_and_save(ia_poisson, ia_erlang, counts_poisson, counts_erlang, lambda_rate, k, T, window, out_dir)
    print("\nГрафики сохранены:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

    print("\nГотово. Открой сохранённые PNG/CSV в папке:", out_dir)


if __name__ == "__main__":
    main()
