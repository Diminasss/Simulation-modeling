import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
from PIL import Image


def image_random(size: int, image_path: str) -> list[float]:
    img: Image.Image = Image.open(image_path).convert("RGB")
    arr: np.ndarray = np.array(img, dtype=np.uint8)

    lsb: np.ndarray = (arr & 1).flatten().astype(np.uint8)

    padded_len: int = (len(lsb) + 7) // 8 * 8
    lsb = np.pad(lsb, (0, padded_len - len(lsb)))

    entropy: np.ndarray = np.packbits(lsb)

    results: list[float] = []

    for counter in range(size):
        counter_bytes: bytes = counter.to_bytes(8, "big")
        h: bytes = hashlib.sha256(entropy.tobytes() + counter_bytes).digest()
        n: int = int.from_bytes(h[:8], "big")
        results.append(n / 2 ** 64)

    return results


def mcg_generator(size: int) -> list[float]:
    a: int = 16807
    c: int = 0
    m: int = 2 ** 31 - 1

    x: int = 1  # seed
    result: list[float] = []

    for _ in range(size):
        x = (a * x + c) % m
        result.append(x / m)

    return result


def fibonacci_generator(size: int, lag1: int = 63, lag2: int = 31) -> list[float]:
    buffer: list[float] = mcg_generator(max(lag1, lag2))
    sequence: list[float] = buffer.copy()

    for i in range(len(buffer), size):
        next_val: float = (sequence[i - lag1] - sequence[i - lag2]) % 1.0
        sequence.append(next_val)

    return sequence


def mt19937_generator(size: int) -> list[float]:
    rng = random.Random()
    return [rng.random() for _ in range(size)]


def basic_stats(data: list[float]) -> str:
    mean: float = float(np.mean(data))
    var: float = float(np.var(data))
    std: float = float(np.std(data))
    return (
        f"Мат. ожидание: {mean:.5f}, "
        f"Дисперсия: {var:.5f}, "
        f"СКО: {std:.5f}"
    )


def save_histograms(
        size: int,
        mcg: list[float],
        fib: list[float],
        mt: list[float],
        ir: list[float]
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))

    axs[0][0].hist(mcg, bins=50, color='skyblue')
    axs[0][0].set_title(f"МКГ, n={size}")

    axs[0][1].hist(fib, bins=50, color='lightgreen')
    axs[0][1].set_title(f"Фибоначчи, n={size}")

    axs[1][0].hist(mt, bins=50, color='salmon')
    axs[1][0].set_title(f"Мерсенн, n={size}")

    axs[1][1].hist(ir, bins=50, color='orange')
    axs[1][1].set_title(f"Картиночный, n={size}")

    plt.tight_layout()
    plt.savefig(f"graphics/histogram_n{size}.png")
    plt.close()


def save_ecdf_plots(
        size: int,
        mcg: list[float],
        fib: list[float],
        mt: list[float],
        ir: list[float]
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))

    axs[0][0].plot(np.sort(mcg), np.linspace(0, 1, len(mcg)), color='blue')
    axs[0][0].set_title(f"ЭФР МКГ, n={size}")

    axs[0][1].plot(np.sort(fib), np.linspace(0, 1, len(fib)), color='green')
    axs[0][1].set_title(f"ЭФР Фибоначчи, n={size}")

    axs[1][0].plot(np.sort(mt), np.linspace(0, 1, len(mt)), color='red')
    axs[1][0].set_title(f"ЭФР Мерсенн, n={size}")

    axs[1][1].plot(np.sort(ir), np.linspace(0, 1, len(ir)), color='orange')
    axs[1][1].set_title(f"ЭФР Картиночный, n={size}")

    plt.tight_layout()
    plt.savefig(f"graphics/ecdf_n{size}.png")
    plt.close()


def save_scatter_diagrams(
        size: int,
        mcg: list[float],
        fib: list[float],
        mt: list[float],
        ir: list[float]
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))

    axs[0][0].scatter(mcg[:-1], mcg[1:], s=2, alpha=0.6, color='blue')
    axs[0][0].set_title(f"МКГ XY, n={size}")

    axs[0][1].scatter(fib[:-1], fib[1:], s=2, alpha=0.6, color='green')
    axs[0][1].set_title(f"Фибоначчи XY, n={size}")

    axs[1][0].scatter(mt[:-1], mt[1:], s=2, alpha=0.6, color='red')
    axs[1][0].set_title(f"Мерсенн XY, n={size}")

    axs[1][1].scatter(ir[:-1], ir[1:], s=2, alpha=0.6, color='orange')
    axs[1][1].set_title(f"Картиночный XY, n={size}")

    plt.tight_layout()
    plt.savefig(f"graphics/scatter_n{size}.png")
    plt.close()


def main() -> int:
    sample_sizes: list[int] = [1_000, 5_000, 10_000, 100_000]

    for size in sample_sizes:
        data_mcg: list[float] = mcg_generator(size)
        data_fib: list[float] = fibonacci_generator(size)
        data_mt: list[float] = mt19937_generator(size)
        data_ir: list[float] = image_random(size, "example_2.jpg")

        print(f"\nАнализ для n = {size}")
        print("МКГ:", basic_stats(data_mcg))
        print("Фибоначчи:", basic_stats(data_fib))
        print("Мерсенн:", basic_stats(data_mt))
        print("Картиночный", basic_stats(data_ir))

        save_histograms(size, data_mcg, data_fib, data_mt, data_ir)
        save_ecdf_plots(size, data_mcg, data_fib, data_mt, data_ir)
        save_scatter_diagrams(size, data_mcg, data_fib, data_mt, data_ir)

    return 0


if __name__ == '__main__':
    main()
