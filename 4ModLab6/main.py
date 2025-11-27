#!/usr/bin/env python3
"""
queue_sim.py

Имитация одноканальной СМО с очередью (FIFO).

Управление через аргументы:
  --arrival {poisson,erlang}    входной поток
  --lambda L                    интенсивность входа (средний поток = L заяв/ед.времени)
  --arrival-k K                 форма эрланга (для arrival="erlang"), целое >=1
  --service {exp,det,erlang}    распределение обслуживания
  --mu M                        интенсивность обслуживания (средний сервисный темп = M заяв/ед.времени)
  --service-k KS                форма эрланга для обслуживания (int)
  --departures N                сколько статистических уходов собрать (после прогрева)
  --warmup W                    сколько первых уходов отбросить (прогрев)
  --seed S                      seed для RNG
  --show-plots                  показать графики (требуется matplotlib)
"""

import argparse
import math
import random
import heapq
import statistics
from scipy import stats
import matplotlib.pyplot as plt

SCIPY_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True


def make_interarrival_sampler(arrival_type: str, lam: float, k: int = 1):
    """Возвращает функцию, генерирующую следующий интервал между прибытием."""
    if arrival_type == "poisson":
        # Интервал экспоненциальный с параметром lam (интенсивность)
        def sampler():
            return random.expovariate(lam)

        return sampler
    elif arrival_type == "erlang":
        # Эрланг(k, rate = k*lam) чтобы средний интервал = 1/lam
        rate = k * lam

        def sampler():
            # сумма k экспоненциальных с rate=rate
            s = 0.0
            for _ in range(k):
                s += random.expovariate(rate)
            return s

        return sampler
    else:
        raise ValueError("Unknown arrival type")


def make_service_sampler(service_type: str, mu: float, k: int = 1):
    """Возвращает функцию, генерирующую время обслуживания одного клиента."""
    if service_type == "exp":
        def sampler():
            return random.expovariate(mu)

        return sampler
    elif service_type == "det":
        def sampler():
            return 1.0 / mu

        return sampler
    elif service_type == "erlang":
        rate = k * mu

        def sampler():
            s = 0.0
            for _ in range(k):
                s += random.expovariate(rate)
            return s

        return sampler
    else:
        raise ValueError("Unknown service type")


class SingleServerQueueSimulator:
    def __init__(self, interarrival_fn, service_fn, seed=None):
        self.interarrival_fn = interarrival_fn
        self.service_fn = service_fn
        if seed is not None:
            random.seed(seed)

    def run(self, target_departures=10000, warmup_departures=1000, verbose=False):
        """
        Запуск имитации.
        target_departures = сколько собрать полезных уходов (после прогрева)
        warmup_departures = сколько первых уходов отбросить
        Возвращает словарь с результатами.
        """
        # События: кортеж (time, type, id)
        # type "arrival" или "departure"
        # используем heapq
        event_queue = []
        now = 0.0

        # запланировать первое прибытие
        t_arrival = now + self.interarrival_fn()
        heapq.heappush(event_queue, (t_arrival, "arrival", None))

        queue = []  # очередь (просто счетчик); FIFO — можем хранить метки
        server_busy = False

        departures_times = []
        total_arrivals = 0
        total_departures = 0
        next_client_id = 0

        while total_departures < (warmup_departures + target_departures):
            if not event_queue:
                # ничего не запланировано — система пуста
                break
            t, ev_type, ev_id = heapq.heappop(event_queue)
            now = t

            if ev_type == "arrival":
                total_arrivals += 1
                client_id = next_client_id
                next_client_id += 1

                # планируем следующее прибытие
                t_next = now + self.interarrival_fn()
                heapq.heappush(event_queue, (t_next, "arrival", None))

                if not server_busy:
                    # сразу в обслуживание
                    service_time = self.service_fn()
                    heapq.heappush(event_queue, (now + service_time, "departure", client_id))
                    server_busy = True
                else:
                    # в очередь
                    queue.append(client_id)

            elif ev_type == "departure":
                total_departures += 1
                departures_times.append(now)

                # если очередь не пуста, взять следующего
                if queue:
                    next_client = queue.pop(0)
                    service_time = self.service_fn()
                    heapq.heappush(event_queue, (now + service_time, "departure", next_client))
                    server_busy = True
                else:
                    server_busy = False

            else:
                raise RuntimeError("Unknown event type")

            # safety: остановка при огромном числе событий
            if total_arrivals > 10_000_000:
                raise RuntimeError("Too many arrivals - check parameters (unstable system?)")

        # отделим прогрев
        useful_departures = departures_times[warmup_departures:]
        result = {
            "departures_all": departures_times,
            "departures_useful": useful_departures,
            "total_arrivals": total_arrivals,
            "total_departures": total_departures,
            "final_time": now
        }
        return result


def analyze_departures(dep_times):
    """Анализ меж-уходных интервалов: среднее, дисперсия, CV, KS-test на экспоненциальность (если scipy)."""
    if len(dep_times) < 2:
        return {"n": len(dep_times)}

    inter = [t2 - t1 for t1, t2 in zip(dep_times[:-1], dep_times[1:])]
    n = len(inter)
    mean = statistics.mean(inter)
    var = statistics.pvariance(inter)  # population variance
    stdev = math.sqrt(var)
    cv = stdev / mean if mean != 0 else float('inf')

    ks_res = None
    if SCIPY_AVAILABLE:
        # тестируем H0: inter ~ Exp(scale=mean)
        # scipy kstest for exponential with cdf = 1 - exp(-x/mean)
        # In scipy, exponential with scale=mean is stats.expon(scale=mean)
        D, pvalue = stats.kstest(inter, 'expon', args=(0, mean))
        ks_res = {"D": float(D), "pvalue": float(pvalue)}
    else:
        ks_res = None

    return {
        "n_intervals": n,
        "mean": mean,
        "var": var,
        "stdev": stdev,
        "cv": cv,
        "ks": ks_res,
        "inter_arrivals": inter
    }


def plot_intervals(inter, show=True, title="Histogram of inter-departure times"):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; cannot plot.")
        return

    # гистограмма + теоретическая плотность экспоненциального распределения с тем же mean
    mean = statistics.mean(inter)
    import numpy as np
    plt.figure(figsize=(8, 4.5))
    plt.hist(inter, bins=50, density=True, alpha=0.6)
    # теоретическая экспонента
    xs = np.linspace(0, max(inter) * 1.05, 200)
    pdf = (1.0 / mean) * np.exp(-xs / mean)
    plt.plot(xs, pdf)  # без указания цвета (по требованию)
    plt.title(title)
    plt.xlabel("inter-departure time")
    plt.ylabel("density")
    plt.grid(True)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate single-server queue with queue (FIFO).")
    parser.add_argument("--arrival", choices=["poisson", "erlang"], default="poisson",
                        help="Тип входного потока")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.9,
                        help="Интенсивность входа (lambda), заявки/единицу времени")
    parser.add_argument("--arrival-k", dest="arrival_k", type=int, default=2,
                        help="k для эрланга (arrival)")
    parser.add_argument("--service", choices=["exp", "det", "erlang"], default="exp",
                        help="Распределение времени обслуживания")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Интенсивность обслуживания (mu) — обратное среднему времени обслуживания")
    parser.add_argument("--service-k", dest="service_k", type=int, default=2,
                        help="k для эрланга (service)")
    parser.add_argument("--departures", type=int, default=20000,
                        help="Сколько уходов собрать после прогрева (полезных)")
    parser.add_argument("--warmup", type=int, default=2000,
                        help="Сколько первых уходов отбросить (прогрев)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для RNG")
    parser.add_argument("--show-plots", action="store_true", help="Показать графики (требуется matplotlib)")
    args = parser.parse_args()

    lam = args.lam
    mu = args.mu

    print("Параметры моделирования:")
    print(f" arrival type: {args.arrival}, lambda={lam}, arrival_k={args.arrival_k}")
    print(f" service type: {args.service}, mu={mu}, service_k={args.service_k}")
    print(f" departures to collect (after warmup): {args.departures}, warmup={args.warmup}")
    print(f" seed={args.seed}")
    print()

    # проверка устойчивости: нагрузка rho = lambda / mu < 1 для single-server average
    rho = lam / mu
    if rho >= 1.0:
        print("Внимание: система неустойчива (rho = lambda/mu >= 1). Результаты будут иметь возрастающую очередь.")
    else:
        print(f"Система устойчивa (rho = {rho:.4f} < 1).")

    interarrival_fn = make_interarrival_sampler(args.arrival, lam, k=max(1, args.arrival_k))
    service_fn = make_service_sampler(args.service, mu, k=max(1, args.service_k))

    sim = SingleServerQueueSimulator(interarrival_fn, service_fn, seed=args.seed)
    res = sim.run(target_departures=args.departures, warmup_departures=args.warmup, verbose=True)

    useful = res["departures_useful"]
    print()
    print(f"Симуляция завершена: собрано {len(useful)} полезных уходов (после прогрева).")
    analysis = analyze_departures(useful)
    print("Анализ меж-уходных интервалов:")
    if "n" in analysis:
        print("  недостаточно данных")
        return
    print(f"  n intervals = {analysis['n_intervals']}")
    print(f"  mean = {analysis['mean']:.6f}")
    print(f"  variance = {analysis['var']:.6f}")
    print(f"  stdev = {analysis['stdev']:.6f}")
    print(f"  CV = {analysis['cv']:.6f}")
    if analysis["ks"] is not None:
        D = analysis["ks"]["D"]
        p = analysis["ks"]["pvalue"]
        print(f"  KS-test vs Exp: D = {D:.6f}, p-value = {p:.6f}")
        if p > 0.05:
            print(
                "    -> нет оснований отвергнуть гипотезу экспоненциальности интервалов ухода (т.е. возможно пуассоновский поток уходов)")
        else:
            print(
                "    -> наблюдаемые интервалы ухода НЕ соответствуют экспоненциальному закону (поток уходов не пуассоновский)")
    else:
        print("  scipy не найден — пропущен KS-test. Установите scipy для выполнения KS-теста.")

    # Если пользователь хочет — показать график
    if args.show_plots:
        plot_intervals(analysis["inter_arrivals"], show=True,
                       title="Inter-departure times histogram (useful departures)")

    # Сохранить простую текстовую сводку в файл
    summary = {
        "params": {
            "arrival": args.arrival,
            "lambda": lam,
            "arrival_k": args.arrival_k,
            "service": args.service,
            "mu": mu,
            "service_k": args.service_k,
            "rho": rho
        },
        "analysis": analysis
    }

    # печать краткого заключения
    print()
    print("Краткое заключение (эвристика):")
    # эвристика: если CV близок к 1 и KS не отвергает -> похоже на экспоненциальное
    cv = analysis["cv"]
    ks_p = analysis["ks"]["pvalue"] if analysis["ks"] is not None else None
    if ks_p is not None and ks_p > 0.05 and abs(cv - 1.0) < 0.15:
        print("  Поток уходов похоже пуассоновский (интервалы ~ экспоненциальны).")
    else:
        print("  Поток уходов НЕ выглядит пуассоновским (интервалы не экспоненциальны).")
    print("  (Сравните для разных комбинаций входного потока и распределения обслуживания.)")

    # напечатать небольшую рекомендацию
    print()
    print("Рекомендация для проверки задания:")
    print(
        " - Запустите симуляцию для сочетаний (arrival=poisson, service=exp) и (arrival=poisson, service=erlang/det).")
    print(
        " - Для M/M/1 (poisson arrivals + exp service) ожидается, что поток уходов — пуассоновский (экспоненциальные меж-уходы).")
    print(
        " - Для M/G/1 (например, service=det или erlang с k>1) поток уходов, вообще говоря, НЕ является пуассоновским.")
    print(" - Меняйте параметры и смотрите KS-test + CV интервалов ухода.")

    # напоследок — возвращаем краткую сводку как печать (можно расширить)
    # (не сохраняем в файл по умолчанию)
    return


if __name__ == "__main__":
    main()
