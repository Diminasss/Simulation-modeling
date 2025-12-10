import numpy as np
import matplotlib.pyplot as plt

# Параметры моделирования (настраиваемые):
arrival_dist = 'poisson'       # 'poisson' или 'erlang' для входного потока
lambda_arrival = 1.0           # интенсивность входного потока λ (для пуассоновского потока)
k_arrival = 3                  # порядок Эрланга (если arrival_dist='erlang')
service_dist = 'exponential'   # 'exponential', 'erlang', 'uniform', 'constant' для обслуживания
lambda_service = 1.2           # параметр λ для распределений экспоненциального/эрланговского обслуживания
k_service = 3                  # порядок Эрланга (если service_dist='erlang')
service_const = 0.8            # фиксированное время обслуживания (если service_dist='constant')
service_uniform = (0.5, 1.5)   # (min, max) для равномерного обслуживания (если 'uniform')

T_end = 100000.0  # полное время моделирования

# Инициализация
t = 0.0                    # текущее время
server_busy = False        # статус сервера (свободен/занят)
next_arrival = 0.0
next_departure = float('inf')
arrival_times = []         # список времен приходов
departure_times = []       # список времен уходов (обслуженных заявок)
lost_count = 0             # счетчик потерянных заявок
served_count = 0           # счетчик обслуженных заявок

# Генерируем время первого прихода
if arrival_dist == 'poisson':
    next_arrival = np.random.exponential(1.0/lambda_arrival)
elif arrival_dist == 'erlang':
    next_arrival = np.sum(np.random.exponential(1.0/(lambda_arrival*k_arrival), size=k_arrival))
else:
    raise ValueError("Unsupported arrival distribution")

# Основной цикл событий
while t < T_end:
    # Определяем ближайшее событие: приход или уход
    if next_arrival <= next_departure:
        # Событие прихода заявки
        t = next_arrival
        if t > T_end:
            break
        arrival_times.append(t)
        if not server_busy:
            # Сервер свободен: начинаем обслуживание этой заявки
            server_busy = True
            served_count += 1
            # Генерируем время обслуживания
            if service_dist == 'exponential':
                service_time = np.random.exponential(1.0/lambda_service)
            elif service_dist == 'erlang':
                service_time = np.sum(np.random.exponential(1.0/(lambda_service*k_service), size=k_service))
            elif service_dist == 'uniform':
                service_time = np.random.uniform(service_uniform[0], service_uniform[1])
            elif service_dist == 'constant':
                service_time = service_const
            else:
                raise ValueError("Unsupported service distribution")
            # Планируем момент ухода (запуск завершится в t+service_time)
            next_departure = t + service_time
        else:
            # Сервер занят: заявка теряется
            lost_count += 1
        # Генерируем следующий интервал до следующего прихода
        if arrival_dist == 'poisson':
            next_arrival = t + np.random.exponential(1.0/lambda_arrival)
        elif arrival_dist == 'erlang':
            next_arrival = t + np.sum(np.random.exponential(1.0/(lambda_arrival*k_arrival), size=k_arrival))
    else:
        # Событие ухода (завершение обслуживания)
        t = next_departure
        if t > T_end:
            break
        departure_times.append(t)
        server_busy = False
        next_departure = float('inf')
        # Без очереди после ухода сервер простаивает до следующего прихода

# Расчёт метрик
total_arrivals = len(arrival_times)
throughput = served_count / T_end
blocking_prob = lost_count / total_arrivals if total_arrivals > 0 else 0.0
utilization = lambda_arrival * (1 - blocking_prob) / lambda_service if lambda_service > 0 else None

print(f"Общее число заявок: {total_arrivals}")
print(f"Обслуженных: {served_count}, Потеряно: {lost_count}")
print(f"Вероятность потери (блокировка): {blocking_prob:.4f}")
print(f"Пропускная способность (интенсивность выхода): {throughput:.4f}")
if utilization is not None:
    print(f"Коэффициент загрузки сервера: {utilization:.4f}")

# Анализ выходного потока
if departure_times:
    departure_intervals = np.diff(departure_times)
    mean_dep = np.mean(departure_intervals)
    var_dep = np.var(departure_intervals)
    print(f"Средний интервал выходного потока: {mean_dep:.4f}, Дисперсия: {var_dep:.4f}")

    # Гистограмма интервалов выходного потока
    plt.figure(figsize=(8,5))
    plt.hist(departure_intervals, bins=30, density=True, alpha=0.7,
             color='skyblue', edgecolor='gray')
    plt.title("Гистограмма интервалов выходного потока")
    plt.xlabel("Интервал между событиями выхода")
    plt.ylabel("Плотность вероятности")
    plt.grid(True)
    plt.savefig("")
    plt.show()

    # (При необходимости можно добавить на график теоретическую плотность,
    #  например экспоненциальную, для наглядного сравнения распределений.)
