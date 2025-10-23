import matplotlib.pyplot as plt
import numpy as np
import random


def mcg_generator(size):
    a, c, m = 16807, 0, 2 ** 31 - 1
    r = 2 ** -52
    result = []
    for _ in range(size):
        r = ((a * r + c) % m) % 1
        result.append(r)
    return result


def fibonacci_generator(size, lag1=63, lag2=31):
    buffer = mcg_generator(max(lag1, lag2))
    sequence = buffer.copy()
    for i in range(len(buffer), size):
        next_val = (sequence[i - lag1] - sequence[i - lag2]) % 1
        sequence.append(next_val)
    return sequence


def mt19937_generator(size):
    rng = random.Random()
    return [rng.random() for _ in range(size)]


def basic_stats(data):
    return (
        f"Мат. ожидание: {np.mean(data):.5f}, "
        f"Дисперсия: {np.var(data):.5f}, "
        f"СКО: {np.std(data):.5f}"
    )


def save_histograms(size, mcg, fib, mt):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(mcg, bins=50, color='skyblue')
    axs[0].set_title(f"МКГ, n={size}")
    axs[1].hist(fib, bins=50, color='lightgreen')
    axs[1].set_title(f"Фибоначчи, n={size}")
    axs[2].hist(mt, bins=50, color='salmon')
    axs[2].set_title(f"Мерсенн, n={size}")
    plt.tight_layout()
    plt.savefig(f"graphics/histogram_n{size}.png")
    plt.close()


def save_ecdf_plots(size, mcg, fib, mt):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(np.sort(mcg), np.linspace(0, 1, len(mcg)), color='blue')
    axs[0].set_title(f"ЭФР МКГ, n={size}")
    axs[1].plot(np.sort(fib), np.linspace(0, 1, len(fib)), color='green')
    axs[1].set_title(f"ЭФР Фибоначчи, n={size}")
    axs[2].plot(np.sort(mt), np.linspace(0, 1, len(mt)), color='red')
    axs[2].set_title(f"ЭФР Мерсенн, n={size}")
    plt.tight_layout()
    plt.savefig(f"graphics/ecdf_n{size}.png")
    plt.close()


def save_scatter_diagrams(size, mcg, fib, mt):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(mcg[:-1], mcg[1:], s=2, alpha=0.6, color='blue')
    axs[0].set_title(f"МКГ XY, n={size}")
    axs[1].scatter(fib[:-1], fib[1:], s=2, alpha=0.6, color='green')
    axs[1].set_title(f"Фибоначчи XY, n={size}")
    axs[2].scatter(mt[:-1], mt[1:], s=2, alpha=0.6, color='red')
    axs[2].set_title(f"Мерсенн XY, n={size}")
    plt.tight_layout()
    plt.savefig(f"graphics/scatter_n{size}.png")
    plt.close()


def save_uniform_plots():
    x = np.linspace(0, 1, 1000)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(x, [1] * len(x), color='black')
    axs[0].set_title("Плотность U(0,1)")
    axs[1].plot(x, x, color='darkred')
    axs[1].set_title("Функция распределения U(0,1)")
    plt.tight_layout()
    plt.savefig("graphics/uniform_plots.png")
    plt.close()


def main() -> int:
    sample_sizes = [1000, 5000, 10000]
    for size in sample_sizes:
        data_mcg = mcg_generator(size)
        data_fib = fibonacci_generator(size)
        data_mt = mt19937_generator(size)

        print(f"\nАнализ для n = {size}")
        print("МКГ:", basic_stats(data_mcg))
        print("Фибоначчи:", basic_stats(data_fib))
        print("Мерсенн:", basic_stats(data_mt))

        save_histograms(size, data_mcg, data_fib, data_mt)
        save_ecdf_plots(size, data_mcg, data_fib, data_mt)
        save_scatter_diagrams(size, data_mcg, data_fib, data_mt)

    save_uniform_plots()
    return 0


if __name__ == '__main__':
    main()
