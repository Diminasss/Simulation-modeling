import numpy as np
import matplotlib.pyplot as plt

# Параметры моделирования (настраиваемые):
arrival_dist = 'poisson'       # 'poisson' или 'erlang' для входного потока
lambda_arrival = 1.0           # интенсивность входного потока λ (для пуассоновского потока)
k_arrival = 3                  # порядок Эрланга (если arrival_dist='erlang')
service_dist = 'exponential'   # 'exponential', 'erlang', 'uniform', 'constant' для обслуживания
lambda_service = 1.2           # параметр λ для экспоненциального/эрланговского обслуживания
k_service = 3                  # порядок Эрланга для обслуживания (если service_dist='erlang')
service_const = 0.8            # фиксированное время обслуживания (если service_dist='constant')
service_uniform = (0.5, 1.5)   # (min, max) для равномерного обслуживания (если service_dist='uniform')
queue_limit = 5              # максимальная ёмкость очереди (число заявок, ожидающих в очереди)

T_end = 100000.0  # полное время моделирования

# Инициализация
t = 0.0                    # текущее время
server_busy = False        # статус сервера (свободен/занят)
next_arrival = 0.0
next_departure = float('inf')
arrival_times = []         # времена прихода заявок
departure_times = []       # времена ухода обслуженных заявок
queue = []                 # очередь ожидающих заявок
lost_count = 0             # счетчик потерянных заявок (при переполнении очереди)
served_count = 0           # счетчик начатых в обслуживании заявок

# Генерация времени первого прихода
if arrival_dist == 'poisson':
    next_arrival = np.random.exponential(1.0 / lambda_arrival)
elif arrival_dist == 'erlang':
    # сумма k_arrival независимых экспоненциальных RV с параметром (lambda_arrival*k_arrival)
    next_arrival = np.sum(np.random.exponential(1.0/(lambda_arrival*k_arrival), size=k_arrival))
else:
    raise ValueError("Unsupported arrival distribution")

# Основной цикл имитации (генерация событий прихода/ухода)
while t < T_end:
    # Определяем, какое событие происходит раньше: приход или уход
    if next_arrival <= next_departure:
        # Событие прихода заявки
        t = next_arrival
        if t > T_end:
            break
        arrival_times.append(t)
        if not server_busy:
            # Сервер свободен -> заявка сразу в обслуживание
            server_busy = True
            served_count += 1
            # Генерация времени обслуживания по выбранному закону
            if service_dist == 'exponential':
                service_time = np.random.exponential(1.0 / lambda_service)
            elif service_dist == 'erlang':
                service_time = np.sum(np.random.exponential(1.0/(lambda_service*k_service), size=k_service))
            elif service_dist == 'uniform':
                service_time = np.random.uniform(service_uniform[0], service_uniform[1])
            elif service_dist == 'constant':
                service_time = service_const
            else:
                raise ValueError("Unsupported service distribution")
            # Планируем событие ухода
            next_departure = t + service_time
        else:
            # Сервер занят -> заявка пытается стать в очередь
            if len(queue) < queue_limit:
                queue.append(t)
            else:
                # Очередь переполнена -> заявка теряется
                lost_count += 1
        # Запланировать следующий приход
        if arrival_dist == 'poisson':
            next_arrival = t + np.random.exponential(1.0 / lambda_arrival)
        elif arrival_dist == 'erlang':
            next_arrival = t + np.sum(np.random.exponential(1.0/(lambda_arrival*k_arrival), size=k_arrival))
    else:
        # Событие ухода (завершение обслуживания)
        t = next_departure
        if t > T_end:
            break
        departure_times.append(t)
        if queue:
            # В очереди есть заявки -> следующая сразу идет в обслуживание
            served_count += 1
            # Извлекаем время прихода (не обязательно для расчётов, но можно сохранить)
            arrival_time = queue.pop(0)
            # Генерация следующего времени обслуживания
            if service_dist == 'exponential':
                service_time = np.random.exponential(1.0 / lambda_service)
            elif service_dist == 'erlang':
                service_time = np.sum(np.random.exponential(1.0/(lambda_service*k_service), size=k_service))
            elif service_dist == 'uniform':
                service_time = np.random.uniform(service_uniform[0], service_uniform[1])
            elif service_dist == 'constant':
                service_time = service_const
            else:
                raise ValueError("Unsupported service distribution")
            next_departure = t + service_time
        else:
            # Очередь пуста -> сервер становится свободным до следующего прихода
            server_busy = False
            next_departure = float('inf')

# Вычисление метрик работы системы
total_arrivals = len(arrival_times)
throughput = served_count / T_end
blocking_prob = lost_count / total_arrivals if total_arrivals > 0 else 0.0
utilization = (lambda_arrival * (1 - blocking_prob) / lambda_service
               if lambda_service > 0 else None)

print(f"Общее число заявок: {total_arrivals}")
print(f"Обслуженных: {served_count}, Потеряно: {lost_count}")
print(f"Вероятность потери (блокировка): {blocking_prob:.4f}")
print(f"Пропускная способность (интенсивность выхода): {throughput:.4f}")
if utilization is not None:
    print(f"Коэффициент загрузки сервера: {utilization:.4f}")

# Анализ выходного потока (интервалы между событиями ухода)
if departure_times:
    departure_intervals = np.diff(departure_times)
    mean_dep = np.mean(departure_intervals)
    var_dep = np.var(departure_intervals)
    print(f"Средний интервал выходного потока: {mean_dep:.4f}, Дисперсия: {var_dep:.4f}")


    plt.hist(departure_intervals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='gray')
    plt.title("Гистограмма интервалов выходного потока")
    plt.xlabel("Интервал между событиями ухода")
    plt.ylabel("Плотность")
    plt.grid(True)
    plt.savefig(".png")
    plt.show()
