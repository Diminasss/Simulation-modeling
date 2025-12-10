"""Программа: СМО с очередью (FIFO) + step-process visualization (N(t) для приходов/начатых/потерянных/уходов).
Сохранена структура оригинала: по 6 сценариев (2 * 3), для каждого строится ОДНА фигура с тремя подграфиками.
Все подписи и сообщения — на русском.
"""
import math
from collections.abc import Callable
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


def my_poisson_stream(lmbda: float, n: int) -> np.ndarray:
    """
    Генерация экспоненциального распределения методом обратной функции (самодельная).

    :param lmbda: float — интенсивность λ (λ > 0)
    :param n: int — количество интервалов (n >= 1)
    :return: np.ndarray — интервалы
    """
    if lmbda <= 0:
        raise ValueError("lmbda must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")

    data: list[float] = []
    for _ in range(n):
        u = np.random.random()  # U ~ Uniform(0,1)
        x = -np.log(1.0 - u) / lmbda
        data.append(x)
    return np.array(data)


def my_erlang_stream(lmbda: float, k: int, n: int) -> np.ndarray:
    """
    Генерация Эрланговского распределения как суммы k экспоненциальных
    с параметром rate = λ * k (такая параметризация согласуется с
    используемой выше реализацией NumPy: mean = 1/λ).
    """
    if lmbda <= 0:
        raise ValueError("lmbda must be > 0")
    if k < 1 or not isinstance(k, int):
        raise ValueError("k must be an integer >= 1")
    if n <= 0:
        raise ValueError("n must be > 0")

    data: list[float] = []
    rate = lmbda * k
    for _ in range(n):
        total = 0.0
        for _ in range(k):
            u = np.random.random()
            x = -np.log(1.0 - u) / rate
            total += x
        data.append(total)
    return np.array(data)


# ----------------------------
# Моделирование и анализ
# ----------------------------
def cumulative_times_from_intervals(intervals: np.ndarray) -> np.ndarray:
    """Преобразовать массив меж-приходных интервалов в моменты времени приходов (кумулятивная сумма)."""
    return np.cumsum(intervals)


def simulate_loss_system(
        arrival_intervals: np.ndarray,
        service_generator: Callable[[], float],
) -> dict:
    """
    Моделирование СМО с очередью (FIFO), один сервер, неограниченная очередь.
    (Имя функции сохранено, чтобы не ломать основной код — теперь она реализует очередь.)
    Возвращаемые поля совместимы со старым кодом, добавлено поле 'start_times' — моменты начала обслуживания.
    """
    arrival_times = cumulative_times_from_intervals(arrival_intervals)
    N = arrival_times.size

    # Результаты
    start_times = np.full(N, np.nan)           # момент начала обслуживания для каждой заявки (если она была обслужена)
    departure_times_all: list[float] = []      # departure times в порядке начала обслуживания
    service_durations_all: list[float] = []    # службы в том же порядке
    accepted_idx: list[int] = []               # индексы заявок, которые были обслужены (в порядке начала обслуживания)
    lost_idx: list[int] = []                   # пустой, т.к. очередь неограничена (оставлено для совместимости)

    queue = deque()  # очередь индексов заявок, ожидающих обслуживания
    server_busy = False
    next_departure = math.inf
    current_time = 0.0
    # если сервер занят — хранится (current_serving_idx, depart_time), но тут достаточно next_departure

    arrivals_idx = 0

    # helper: начать обслуживание заявки idx в момент t
    def start_service(idx: int, t: float):
        s = float(service_generator())
        start_times[idx] = float(t)
        depart = float(t + s)
        departure_times_all.append(depart)
        service_durations_all.append(s)
        accepted_idx.append(idx)
        return depart

    # основной цикл событий: события — приходы и ухода (уходы всегда обрабатываем до приходов в тот же момент)
    while arrivals_idx < N or server_busy:
        next_arrival = arrival_times[arrivals_idx] if arrivals_idx < N else math.inf

        if not server_busy:
            # сервер свободен: если есть заявка в очереди — обслуживаем её немедленно
            if queue:
                q_idx = queue.popleft()
                depart = start_service(q_idx, current_time)
                next_departure = depart
                server_busy = True
                continue
            # иначе если есть приход — начинаем его немедленно
            if next_arrival < math.inf:
                current_time = next_arrival
                depart = start_service(arrivals_idx, current_time)
                next_departure = depart
                server_busy = True
                arrivals_idx += 1
                continue
            # ничего не осталось
            break
        else:
            # сервер занят: смотрим, что раньше — приход или уход
            if next_arrival <= next_departure:
                # приход до ухода (или в тот же момент) — ставим в очередь (FIFO)
                current_time = next_arrival
                queue.append(arrivals_idx)
                # отметим статус как "queued" позже при анализе (совместимость со старой логикой)
                arrivals_idx += 1
                continue
            else:
                # уход ранее, обрабатываем уход
                current_time = next_departure
                # после ухода — если есть очередь, начинаем следующую заявку в момент ухода
                if queue:
                    q_idx = queue.popleft()
                    depart = start_service(q_idx, current_time)
                    next_departure = depart
                    server_busy = True
                    continue
                else:
                    # очередь пуста — сервер стал свободен
                    server_busy = False
                    next_departure = math.inf
                    continue

    # конвертируем в numpy массивы, departure_times уже в порядке начала обслуживания
    departure_times_arr = np.array(departure_times_all)
    service_durations_arr = np.array(service_durations_all)

    # определим lost_idx (в этой реализации никто не теряется, но поле оставляем)
    # определим accepted_idx — уже содержит индексы обслуженных заявок в порядке начала обслуживания
    # чтобы сохранить совместимость с analyze_and_plot_results, вернём accepted_indices как список
    total_time = max(arrival_times[-1], departure_times_arr[-1] if departure_times_arr.size > 0 else 0.0)

    return {
        "arrival_times": arrival_times,
        "accepted_indices": accepted_idx,
        "lost_indices": lost_idx,
        "departure_times": departure_times_arr,
        "service_durations": service_durations_arr,
        "start_times": start_times,   # новое поле: моменты начала обслуживания (NaN для не обслуженных)
        "total_time": total_time,
    }


def make_service_generator(
        kind: str,
        *,
        rate: float | None = None,
        deterministic_time: float | None = None,
        erlang_k: int | None = None,
) -> Callable[[], float]:
    """
    Построить генератор времен обслуживания заданного типа.
    """
    if kind == "exp":
        if rate is None or rate <= 0:
            raise ValueError("rate must be provided and > 0 for exponential service")

        def gen() -> float:
            return np.random.exponential(1.0 / rate)

        return gen

    if kind == "deterministic":
        if deterministic_time is None or deterministic_time <= 0:
            raise ValueError("deterministic_time must be provided and > 0 for deterministic service")

        def gen() -> float:
            return float(deterministic_time)

        return gen

    if kind == "erlang":
        if rate is None or rate <= 0:
            raise ValueError("rate must be provided and > 0 for erlang service")
        if erlang_k is None or erlang_k < 1:
            raise ValueError("erlang_k must be integer >= 1 for erlang service")

        def gen() -> float:
            return float(my_erlang_stream(rate, erlang_k, 1)[0])

        return gen

    raise ValueError("Unknown service kind")


def make_step_process(event_times: np.ndarray) -> tuple[list[float], list[int]]:
    """
    Преобразует список моментов событий в ступенчатый график N(t):
    возвращает xs и ys, подходящие для plt.step(..., where='post').
    """
    if event_times.size == 0:
        return [0.0], [0]
    t_sorted = np.sort(event_times)
    xs: list[float] = [0.0]
    ys: list[int] = [0]
    count = 0
    for t in t_sorted:
        xs.append(float(t)); ys.append(count)
        count += 1
        xs.append(float(t)); ys.append(count)
    return xs, ys


def analyze_and_plot_results(
        results: dict,
        * ,
        title_suffix: str = ""
) -> dict:
    """
    Проанализировать результаты моделирования и вывести графики (три подграфика).
    Подписи и печать — на русском.
    Поддерживает поле 'start_times' в results (если есть — используется для подсчёта ожиданий).
    """
    arrival_times: np.ndarray = results["arrival_times"]
    departure_times: np.ndarray = results["departure_times"]
    accepted_idx = results["accepted_indices"]
    lost_idx = results["lost_indices"]
    start_times = results.get("start_times", None)
    total_time: float = results["total_time"]

    arrivals_count = arrival_times.size
    accepted_count = len(accepted_idx)
    lost_count = len(lost_idx)
    loss_prob = lost_count / arrivals_count if arrivals_count > 0 else float("nan")
    rate_in = arrivals_count / total_time if total_time > 0 else float("nan")
    rate_out = accepted_count / total_time if total_time > 0 else float("nan")

    # вычислим времена начала обслуживаний для accepted:
    if start_times is not None and accepted_count > 0:
        # accepted_idx содержит индексы заявок в порядке их начала обслуживания
        start_times_for_accepted = np.array([start_times[i] for i in accepted_idx])
        durations = departure_times - start_times_for_accepted
        total_busy = float(np.sum(durations))
        utilization = total_busy / total_time if total_time > 0 else float("nan")
    elif accepted_count > 0:
        # fallback — старая совместимость: считаем, что start = arrival_time for accepted
        start_times_for_accepted = arrival_times[np.array(accepted_idx)]
        durations = departure_times - start_times_for_accepted
        total_busy = float(np.sum(durations))
        utilization = total_busy / total_time if total_time > 0 else float("nan")
    else:
        total_busy = 0.0
        utilization = 0.0
        durations = np.array([])

    # меж-уходы
    if departure_times.size >= 2:
        inter_departures = np.diff(np.sort(departure_times))
    else:
        inter_departures = np.array([])

    mean_id = float(np.mean(inter_departures)) if inter_departures.size > 0 else float("nan")
    var_id = float(np.var(inter_departures, ddof=1)) if inter_departures.size > 1 else float("nan")
    cv_id = (math.sqrt(var_id) / mean_id) if (not math.isnan(mean_id) and mean_id > 0) else float("nan")

    # counts per window
    window = max(1.0, mean_id * 5 if not math.isnan(mean_id) else 1.0)
    edges = np.arange(0.0, max(total_time, departure_times[-1] if departure_times.size > 0 else 0.0) + window, window)
    counts_per_window, _ = np.histogram(departure_times, bins=edges)
    mean_counts = float(np.mean(counts_per_window)) if counts_per_window.size > 0 else float("nan")
    var_counts = float(np.var(counts_per_window, ddof=1)) if counts_per_window.size > 1 else float("nan")

    # времена ожидания: если есть start_times — вычислим среднее ожидание (start - arrival)
    if start_times is not None and accepted_count > 0:
        start_arr = np.array([start_times[i] for i in accepted_idx])
        wait_times = start_arr - arrival_times[np.array(accepted_idx)]
        mean_wait = float(np.mean(wait_times)) if wait_times.size > 0 else float("nan")
        var_wait = float(np.var(wait_times, ddof=1)) if wait_times.size > 1 else float("nan")
    else:
        mean_wait = float("nan")
        var_wait = float("nan")

    # Печать основной сводки (русский)
    print("=== РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ===")
    print(f"Общее время моделирования = {total_time:.6f}")
    print(f"Приходы: {arrivals_count}, обслужено: {accepted_count}, потеряно: {lost_count}")
    print(f"Вероятность потерь = {loss_prob:.6f}")
    print(f"Интенсивность поступления (оценка) = {rate_in:.6f}, пропускная способность = {rate_out:.6f}, загрузка = {utilization:.6f}")
    print("---- меж-уходы ----")
    print(f"среднее = {mean_id:.6f}, дисперсия = {var_id:.6f}, CV = {cv_id:.6f}")
    print("---- counts per window ----")
    print(f"среднее count/окно = {mean_counts:.6f}, дисперсия = {var_counts:.6f}")
    if not math.isnan(mean_counts) and mean_counts > 0:
        print(f"Var/Mean (counts) = {var_counts / mean_counts:.3f}  (≈1 -> пуассоноподобно)")
    if not math.isnan(mean_wait):
        print(f"Среднее время ожидания = {mean_wait:.6f}, дисперсия ожидания = {var_wait:.6f}")
    print("===============================")

    # Построение графиков (3 подграфика, как в оригинале)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    ax0, ax1, ax2 = axes

    # Step-processes (N(t))
    accepted_times = start_times_for_accepted if (start_times is not None and accepted_count > 0) else (arrival_times[np.array(accepted_idx)] if accepted_count > 0 else np.array([]))
    lost_times = arrival_times[np.array(lost_idx)] if lost_count > 0 else np.array([])
    dep_times = departure_times if departure_times.size > 0 else np.array([])

    arr_x, arr_y = make_step_process(arrival_times)
    acc_x, acc_y = make_step_process(accepted_times)
    lost_x, lost_y = make_step_process(lost_times)
    dep_x, dep_y = make_step_process(dep_times)

    ax0.step(arr_x, arr_y, where="post", label=f"Приходы N_a(t) = {arrivals_count}", linewidth=1.2)
    ax0.step(acc_x, acc_y, where="post", label=f"Начало обслуживания N_s(t) = {accepted_times.size if accepted_times is not None else 0}", linewidth=1.2)
    if lost_times.size > 0:
        ax0.step(lost_x, lost_y, where="post", label=f"Потери N_l(t) = {lost_times.size}", linewidth=1.2)
    ax0.step(dep_x, dep_y, where="post", label=f"Уходы N_d(t) = {dep_times.size}", linewidth=1.2)

    ax0.set_xlabel("время")
    ax0.set_ylabel("число")
    ax0.set_title(f"Ступенчатые процессы N(t) {title_suffix}")
    ax0.grid(True)
    ax0.legend(loc="upper left")

    # Гистограмма меж-уходов
    if inter_departures.size > 0:
        bins = min(100, max(10, int(np.sqrt(inter_departures.size))))
        ax1.hist(inter_departures, bins=bins, density=True, alpha=0.6, label="эмпирические меж-уходы")
        lambda_hat = 1.0 / mean_id if mean_id > 0 and not math.isnan(mean_id) else None
        if lambda_hat:
            xs = np.linspace(0, np.max(inter_departures) * 1.1, 200)
            ax1.plot(xs, lambda_hat * np.exp(-lambda_hat * xs), label=f"экспоненциальная подгонка (λ={lambda_hat:.3f})")
        ax1.set_title("Гистограмма меж-уходов")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "Недостаточно уходов для гистограммы меж-уходов", ha="center")
        ax1.set_axis_off()

    # Counts per window + Poisson overlay
    if counts_per_window.size > 0:
        max_k = max(10, int(np.max(counts_per_window)))
        ax2.hist(counts_per_window, bins=range(0, max_k + 2), density=False, alpha=0.6, align="left",
                 label="число уходов в окне")
        if not math.isnan(mean_counts) and mean_counts > 0:
            xs = np.arange(0, max_k + 1)
            pois_pmf = np.array([math.exp(-mean_counts) * (mean_counts ** k) / math.factorial(k) for k in xs])
            scale = counts_per_window.size
            ax2.plot(xs, pois_pmf * scale, "o-", label=f"Пуассон(среднее={mean_counts:.3f})")
        ax2.set_title(f"Распределение числа уходов за окно (окно={window:.3f})")
        ax2.set_xlabel("число уходов в окне")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Нет окон для анализа counts per window", ha="center")
        ax2.set_axis_off()

    plt.tight_layout()
    try:
        plt.savefig(f"results/{title_suffix}.png")
    except Exception:
        pass
    plt.show()

    metrics = {
        "arrivals_count": arrivals_count,
        "accepted_count": accepted_count,
        "lost_count": lost_count,
        "loss_prob": loss_prob,
        "rate_in": rate_in,
        "rate_out": rate_out,
        "utilization": utilization,
        "total_busy": total_busy,
        "mean_inter_departure": mean_id,
        "var_inter_departure": var_id,
        "cv_inter_departure": cv_id,
        "mean_counts_per_window": mean_counts,
        "var_counts_per_window": var_counts,
        "mean_wait": mean_wait,
        "var_wait": var_wait,
    }
    return metrics


# ----------------------------
# Демонстрация: несколько сценариев
# ----------------------------
def main():
    """
    Демонстрация работы: прогон нескольких сценариев (poisson/erlang arrivals)
    и разных законов обслуживания (exp/deterministic/erlang).
    Количество сценариев и комбинаций сохранено — получится 2 * 3 = 6 фигур.
    """
    np.random.seed(42)

    # Параметры моделирования
    n_arrivals = 500000
    arrival_scenarios = [
        ("poisson", {"lambda": 0.9}),
        ("erlang_k3", {"lambda": 0.9, "k": 3}),
    ]
    service_scenarios = [
        ("exp", {"rate": 1.2}),
        ("deterministic", {"deterministic_time": 1.0}),
        ("erlang", {"rate": 1.2, "k": 3}),
    ]

    for arr_name, arr_params in arrival_scenarios:
        if arr_name == "poisson":
            ia = my_poisson_stream(arr_params["lambda"], n_arrivals)
            arrival_descr = f"Пуассон(λ={arr_params['lambda']})"
        else:
            ia = my_erlang_stream(arr_params["lambda"], arr_params["k"], n_arrivals)
            arrival_descr = f"Эрланг(k={arr_params['k']}, λ={arr_params['lambda']})"

        for svc_kind, svc_params in service_scenarios:
            if svc_kind == "exp":
                gen = make_service_generator("exp", rate=svc_params["rate"])
                svc_descr = f"Экспоненциальное(rate={svc_params['rate']})"
            elif svc_kind == "deterministic":
                gen = make_service_generator("deterministic", deterministic_time=svc_params["deterministic_time"])
                svc_descr = f"Детерминированное(t={svc_params['deterministic_time']})"
            elif svc_kind == "erlang":
                gen = make_service_generator("erlang", rate=svc_params["rate"], erlang_k=svc_params["k"])
                svc_descr = f"Эрланг(k={svc_params['k']}, rate={svc_params['rate']})"
            else:
                continue

            print("\n\n=====================================")
            print(f"Сценарий: приход = {arrival_descr}   обслуживание = {svc_descr}")
            results = simulate_loss_system(ia, gen)
            metrics = analyze_and_plot_results(results, title_suffix=f"[Arr={arr_name},Svc={svc_kind}]")

            # простое эвристическое заключение о типе выходного потока
            mean_counts = metrics["mean_counts_per_window"]
            var_counts = metrics["var_counts_per_window"]
            var_mean_ratio = var_counts / mean_counts if mean_counts and not math.isnan(mean_counts) else float("nan")
            if not math.isnan(var_mean_ratio):
                if abs(var_mean_ratio - 1.0) < 0.1:
                    conclusion = "дисперсия ≈ среднее -> выходной поток пуассоноподобен (по counts)"
                else:
                    conclusion = "дисперсия != среднее -> выходной поток НЕ пуассоноподобен"
            else:
                conclusion = "недостаточно данных для вывода о пуассоноподобии"
            print("Авто-вывод:", conclusion)


if __name__ == "__main__":
    main()
