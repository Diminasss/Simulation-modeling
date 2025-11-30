import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# Метрики потока
# ---------------------------------------------------------------
@dataclass
class StreamMetrics:
    """
    Структура для хранения статистических метрик потока событий.

    Attributes:
        mean (float): математическое ожидание интервалов
        variance (float): дисперсия интервалов
        std (float): среднеквадратичное отклонение
        coef_variation (float): коэффициент вариации (std / mean)
    """
    mean: float
    variance: float
    std: float
    coef_variation: float

    def __str__(self) -> str:
        return (f"Мат. ожидание: {self.mean:.5f}\n"
                f"Дисперсия: {self.variance:.5f}\n"
                f"СКО: {self.std:.5f}\n"
                f"Коэф. вариации: {self.coef_variation:.5f}\n")


# ---------------------------------------------------------------
# Стандартные реализации (NumPy)
# ---------------------------------------------------------------
def poisson_stream(lmbda: float, n: int) -> np.ndarray:
    """
    Генерация пуассоновского потока с использованием экспоненциального распределения (NumPy).

    :param lmbda: float — интенсивность λ (λ > 0)
    :param n: int — количество интервалов (n >= 1)
    :return: np.ndarray — интервалы потока
    """
    if lmbda <= 0:
        raise ValueError("lmbda must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")
    return np.random.exponential(1 / lmbda, size=n)


def erlang_stream(lmbda: float, k: int, n: int) -> np.ndarray:
    """
    Эрланговский поток с использованием NumPy Gamma с параметризацией,
    обеспечивающей среднее 1/λ (scale = 1/(λ*k), shape = k).

    :param lmbda: float — интенсивность λ (λ > 0)
    :param k: int — порядок Эрланга (k >= 1)
    :param n: int — количество интервалов (n >= 1)
    :return: np.ndarray — интервалы потока
    """
    if lmbda <= 0:
        raise ValueError("lmbda must be > 0")
    if k < 1 or not isinstance(k, int):
        raise ValueError("k must be an integer >= 1")
    if n <= 0:
        raise ValueError("n must be > 0")
    return np.random.gamma(shape=k, scale=1 / (lmbda * k), size=n)


# ---------------------------------------------------------------
# Пользовательские реализации (самодельные)
# ---------------------------------------------------------------
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
        # Инверсный метод: X = -ln(1 - U) / λ
        x = -np.log(1.0 - u) / lmbda
        data.append(x)
    return np.array(data)


def my_erlang_stream(lmbda: float, k: int, n: int) -> np.ndarray:
    """
    Генерация Эрланговского распределения как суммы k экспоненциальных
    с параметром rate = λ * k (такая параметризация согласуется со
    используемой выше реализацией NumPy: mean = 1/λ).

    :param lmbda: float — интенсивность λ (λ > 0)
    :param k: int — порядок Эрланга (k >= 1)
    :param n: int — количество интервалов (n >= 1)
    :return: np.ndarray — интервалы
    """
    if lmbda <= 0:
        raise ValueError("lmbda must be > 0")
    if k < 1 or not isinstance(k, int):
        raise ValueError("k must be an integer >= 1")
    if n <= 0:
        raise ValueError("n must be > 0")

    data: list[float] = []
    rate = lmbda * k  # выбираем rate так, чтобы mean = 1/λ (см. обоснование выше)

    for _ in range(n):
        total = 0.0
        for _ in range(k):
            u = np.random.random()
            x = -np.log(1.0 - u) / rate
            total += x
        data.append(total)

    return np.array(data)


# ---------------------------------------------------------------
# Вычисление метрик
# ---------------------------------------------------------------
def get_metrics(data: np.ndarray) -> StreamMetrics:
    """
    Расчет статистических характеристик потока.

    :param data: np.ndarray — интервалы между событиями
    :return: StreamMetrics — вычисленные метрики
    """
    if data.size == 0:
        raise ValueError("data must contain at least one element")
    mean = float(np.mean(data))
    variance = float(np.var(data))
    std = float(np.std(data))
    cv = std / mean
    return StreamMetrics(mean, variance, std, cv)


# ---------------------------------------------------------------
# Построение распределений
# ---------------------------------------------------------------
def plot_distributions(d1: np.ndarray, d2: np.ndarray, k: int,
                       label1: str = "Поток 1", label2: str = "Поток 2") -> None:
    """
    Графическое сравнение двух распределений.

    :param d1: np.ndarray — первый поток
    :param d2: np.ndarray — второй поток
    :param k: int — порядок Эрланга (используется в подписи)
    :param label1: str — подпись первого графика
    :param label2: str — подпись второго графика
    :return: None
    """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.hist(d1, bins=50, alpha=0.75)
    plt.title(f"Распределение {label1}")
    plt.xlabel("Интервал")
    plt.ylabel("Частота")

    plt.subplot(1, 2, 2)
    plt.hist(d2, bins=50, alpha=0.75)
    plt.title(f"Распределение {label2} (k={k})")
    plt.xlabel("Интервал")
    plt.ylabel("Частота")

    plt.tight_layout()
    plt.savefig(f"results/Гистограмма для {label1} и {label2}.png")
    plt.show()


# ---------------------------------------------------------------
# Текстовое сравнение потоков
# ---------------------------------------------------------------
def compare_streams(name1: str, m1: StreamMetrics,
                    name2: str, m2: StreamMetrics) -> None:
    """
    Текстовое сравнение двух потоков: выводит метрики и краткие выводы.

    :param name1: str — название 1-го потока
    :param m1: StreamMetrics — метрики первого потока
    :param name2: str — название 2-го потока
    :param m2: StreamMetrics — метрики второго потока
    :return: None
    """
    print(f"\n===== СРАВНЕНИЕ {name1.upper()} vs {name2.upper()} =====\n")
    print(name1 + ":\n" + str(m1))
    print(name2 + ":\n" + str(m2))
    print("Итоги:")
    print("→ Меньшее среднее время:", name1 if m1.mean < m2.mean else name2)
    print("→ Более регулярный поток:", name1 if m1.coef_variation < m2.coef_variation else name2)


# ---------------------------------------------------------------
# Анализаторы — разделение по сигнатурам функций
# ---------------------------------------------------------------
def analyze_poisson(poissonFunc1, poissonFunc2,
                    lmbda: float, n: int,
                    name1: str, name2: str) -> tuple[StreamMetrics, StreamMetrics]:
    """
    Анализ и сравнение двух реализаций Пуассоновского потока.
    Обе функции должны иметь сигнатуру func(lmbda: float, n: int).

    :param poissonFunc1: callable — первая функция генерации
    :param poissonFunc2: callable — вторая функция генерации
    :param lmbda: float — интенсивность λ
    :param n: int — количество интервалов
    :param name1: str — подпись первой реализации
    :param name2: str — подпись второй реализации
    :return: tuple(StreamMetrics, StreamMetrics)
    """
    data1 = poissonFunc1(lmbda, n)
    data2 = poissonFunc2(lmbda, n)

    m1 = get_metrics(data1)
    m2 = get_metrics(data2)

    compare_streams(name1, m1, name2, m2)
    # Для Пуассона параметр k не важен — передаём 1 для подписи
    plot_distributions(data1, data2, k=1, label1=name1, label2=name2)

    return m1, m2


def analyze_erlang(erlangFunc1, erlangFunc2,
                   lmbda: float, k: int, n: int,
                   name1: str, name2: str) -> tuple[StreamMetrics, StreamMetrics]:
    """
    Анализ и сравнение двух реализаций Эрланговского потока.
    Обе функции должны иметь сигнатуру func(lmbda: float, k: int, n: int).

    :param erlangFunc1: callable — первая функция генерации
    :param erlangFunc2: callable — вторая функция генерации
    :param lmbda: float — интенсивность λ
    :param k: int — порядок Эрланга
    :param n: int — количество интервалов
    :param name1: str — подпись первой реализации
    :param name2: str — подпись второй реализации
    :return: tuple(StreamMetrics, StreamMetrics)
    """
    data1 = erlangFunc1(lmbda, k, n)
    data2 = erlangFunc2(lmbda, k, n)

    m1 = get_metrics(data1)
    m2 = get_metrics(data2)

    compare_streams(name1, m1, name2, m2)
    plot_distributions(data1, data2, k=k, label1=name1, label2=name2)

    return m1, m2


# ---------------------------------------------------------------
# Пример использования — основной блок
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Параметры моделирования
    n = 5000
    lmbda = 3.0
    k = 3

    # Сравнение пользовательской и NumPy-реализаций для Пуассона
    myPoissonMetrics, numpyPoissonMetrics = analyze_poisson(
        my_poisson_stream, poisson_stream,
        lmbda, n,
        "Мой Пуассон", "NumPy Пуассон"
    )

    # Сравнение пользовательской и NumPy-реализаций для Эрланга
    myErlangMetrics, numpyErlangMetrics = analyze_erlang(
        my_erlang_stream, erlang_stream,
        lmbda, k, n,
        "Мой Эрланг", "NumPy Эрланг"
    )
