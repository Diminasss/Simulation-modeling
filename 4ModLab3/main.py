import random
import time
import statistics
import math


def estimate_pi(n_points: int) -> float:
    """Оценивает число π методом Монте-Карло."""
    inside = 0
    for _ in range(n_points):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / n_points


def main():
    print("=== Определение числа π методом Монте-Карло ===")
    total_trials = 100       # количество независимых экспериментов
    max_points = 200_000     # максимум бросков в одном эксперименте
    step = 2000              # шаг увеличения числа бросков
    eps = 0.009              # допуск на погрешность (±0.005 соответствует второму знаку)

    random.seed(228)          # фиксируем генератор для воспроизводимости

    start_time = time.time()
    results = []

    for _ in range(total_trials):
        found = None
        for n in range(step, max_points + step, step):
            pi_est = estimate_pi(n)
            # Проверяем достижение точности до двух знаков
            if abs(pi_est - math.pi) < eps:
                found = n
                break
        results.append(found)

    duration = time.time() - start_time
    successful = [r for r in results]

    print(f"\nЭкспериментов: {total_trials}, максимальное число бросков: {max_points}")
    print(f"Время выполнения: {duration:.2f} с")

    if successful:
        success_rate = len(successful) / total_trials
        median_n = statistics.median(successful)
        mean_n = statistics.mean(successful)

        print(
            f"Точность до второго знака (3.14) достигнута в {len(successful)} из {total_trials} случаев "
            f"({success_rate:.2%})"
        )
        print(f"Медиана количества бросков: {int(median_n)}")
        print(f"Среднее количество бросков: {int(mean_n)}")
    else:
        print("Ни в одном эксперименте точность не достигнута.")

    # Исправленная теоретическая формула
    theoretical_n = (math.pi * (4 - math.pi)) / (eps ** 2)
    print(f"\nТеоретически для точности ±{eps} нужно примерно {int(theoretical_n)} бросков")


if __name__ == "__main__":
    main()
