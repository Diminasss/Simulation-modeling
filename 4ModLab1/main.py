"""
Игра «Жизнь»
--------------------------------------------------

Описание:
    Это классический клеточный автомат, в котором каждая клетка живёт, умирает или рождается
    в зависимости от количества соседей.

Основные параметры:
    - Правила (B/S) — Birth/Survival. Например, классические "B3/S23" или варианты вроде "B36/S23".
    - Тор (wrap=True) — поле замыкается по краям, т.е. верхняя граница соединяется с нижней.
    - Можно вставлять готовые шаблоны (глайдер, пушка Гозпера и т.д.)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Optional, Set, Any


class GameOfLife:
    """
       Класс для моделирования клеточного автомата «Игра Жизнь».

       Атрибуты:
        :param rows (int): количество строк в сетке
        :param cols (int): количество столбцов в сетке
        :param wrap (bool): замыкание по краям (тор)
        :param rule_birth (set[int]): набор чисел соседей, при которых рождается новая клетка (B)
        :param rule_survive (set[int]): набор чисел соседей, при которых клетка выживает (S)
        :param grid (np.ndarray): текущее состояние поля (0 — мёртвая клетка, 1 — живая)
       """

    def __init__(self, size: Tuple[int, int] = (60, 60), wrap: bool = True, rule_b: str = "3",
                 rule_s: str = "23") -> None:
        """
        Инициализация игры.

        Args:
        :param size: размеры сетки (строки, столбцы)
        :param wrap: если True — поле замыкается по краям (тор)
        :param rule_b: строка, задающая правила рождения (Birth)
        :param rule_s: строка, задающая правила выживания (Survival)
        """
        self.rows: int
        self.cols: int
        self.rows, self.cols = size
        self.wrap: bool = wrap

        # Преобразуем строки правил в множества чисел
        self.rule_birth: Set[int] = {int(ch) for ch in rule_b}
        self.rule_survive: Set[int] = {int(ch) for ch in rule_s}

        # Инициализируем пустое поле
        self.grid: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.uint8)

    # ---------------------------------------------------------

    def randomize(self, p: float = 0.2, seed: Optional[int] = None) -> None:
        """
        Случайно заполняет поле живыми клетками с вероятностью p.

        Args:
        :param p: вероятность рождения живой клетки (0.0–1.0)
        :param seed: случайное зерно (для воспроизводимости)
        """
        if seed is not None:
            np.random.seed(seed)
        self.grid = (np.random.random((self.rows, self.cols)) < p).astype(np.uint8)

    # ---------------------------------------------------------

    def set_pattern(self, top_left: Tuple[int, int], pattern: np.ndarray) -> None:
        """
        Вставляет заданный шаблон в сетку по указанным координатам.

        Args:
        :param top_left: координаты верхнего левого угла вставки (строка, столбец)
        :param pattern: двумерный массив numpy с 0 и 1 (мёртвые/живые клетки)
        """
        r0, c0 = top_left
        pr, pc = pattern.shape
        r1, c1 = min(self.rows, r0 + pr), min(self.cols, c0 + pc)
        self.grid[r0:r1, c0:c1] = pattern[:r1 - r0, :c1 - c0]

    # ---------------------------------------------------------

    def _count_neighbors(self) -> np.ndarray:
        """
        Подсчитывает количество живых соседей для каждой клетки.

        Returns:
        :return: Массив такого же размера, где каждая ячейка содержит число соседей (0–8).
        """
        if self.wrap:
            # Сдвиги в 8 направлений (все соседи)
            n = (
                    np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
                    np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) +
                    np.roll(np.roll(self.grid, 1, axis=0), 1, axis=1) +
                    np.roll(np.roll(self.grid, 1, axis=0), -1, axis=1) +
                    np.roll(np.roll(self.grid, -1, axis=0), 1, axis=1) +
                    np.roll(np.roll(self.grid, -1, axis=0), -1, axis=1)
            )
            return n
        else:
            # Границы считаются мёртвыми
            padded = np.pad(self.grid, pad_width=1, mode="constant", constant_values=0)
            n = np.zeros_like(self.grid, dtype=np.uint8)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    n += padded[1 + dr:self.rows + 1 + dr, 1 + dc:self.cols + 1 + dc]
            return n

    # ---------------------------------------------------------

    def step(self) -> None:
        """
        Делает один шаг эволюции (одно поколение) в соответствии с правилами B/S.
        """
        n = self._count_neighbors()

        # Новое поколение: рождаются или выживают
        birth = ((self.grid == 0) & np.isin(n, list(self.rule_birth))).astype(np.uint8)
        survive = ((self.grid == 1) & np.isin(n, list(self.rule_survive))).astype(np.uint8)
        self.grid = (birth | survive).astype(np.uint8)

    # ---------------------------------------------------------

    def run_console(self, steps: int = 10, pause: float = 0.5) -> None:
        """
        Запускает простую текстовую анимацию в консоли.

        Args:
        :param steps: количество поколений
        :param pause: задержка между кадрами (в секундах)
        """
        import time, os, platform

        clear_cmd: str = "cls" if platform.system().lower().startswith("win") else "clear"

        for i in range(steps):
            os.system(clear_cmd)
            print(f"Поколение {i + 1}:")
            for row in range(self.rows):
                print(''.join('█' if v else ' ' for v in self.grid[row]))
            self.step()
            time.sleep(pause)

    # ---------------------------------------------------------

    def animate(self, frames: int = 200, interval: int = 80, cmap: Optional[str] = "plasma", save: bool = False,
                save_path: str = "game_of_life.gif") -> animation.FuncAnimation:
        """
        Показывает анимацию с помощью matplotlib.

        Args:
        :param frames: количество кадров (поколений)
        :param interval: задержка между кадрами (мс)
        :param cmap: цветовая карта matplotlib (по умолчанию — plasma)
        :param save: если True — сохранить анимацию в файл
        :param save_path: путь к файлу для сохранения

        Returns:
        :return: Объект matplotlib.animation.FuncAnimation
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(self.grid, interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Game of Life — правила B{''.join(map(str, sorted(self.rule_birth)))} / S{''.join(map(str, sorted(self.rule_survive)))}"
        )

        def update(_frame: int) -> tuple[Any, ...]:
            self.step()
            im.set_data(self.grid)
            return (im,)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        plt.close(fig)  # предотвращает двойной вывод

        if save:
            try:
                ani.save(save_path, writer="pillow", fps=1000 / interval)
                print(f"Анимация сохранена: {save_path}")
            except Exception as e:
                print("Ошибка при сохранении анимации:", e)
        return ani


# ---------------------------------------------------------
# Готовые шаблоны (узоры)
# ---------------------------------------------------------

def glider() -> np.ndarray:
    """
    :return: Возвращает шаблон 'глайдера' (движущаяся структура).
    """
    return np.array([[0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1]], dtype=np.uint8)


def gosper_glider_gun() -> np.ndarray:
    """
    :return: Возвращает шаблон пушки Гозпера — бесконечно выпускает глайдеры.
    """
    gun = np.zeros((9, 36), dtype=np.uint8)
    coords = [
        (5, 1), (5, 2), (6, 1), (6, 2),
        (3, 13), (3, 14), (4, 12), (4, 16), (5, 11), (5, 17),
        (6, 11), (6, 15), (6, 17), (6, 18),
        (7, 13), (7, 14),
        (1, 25), (2, 23), (2, 25), (3, 21), (3, 22),
        (4, 21), (4, 22), (5, 21), (5, 22),
        (6, 23), (6, 25), (7, 25),
        (3, 35), (3, 36), (4, 35), (4, 36)
    ]
    for r, c in coords:
        if 0 <= r - 1 < gun.shape[0] and 0 <= c - 1 < gun.shape[1]:
            gun[r - 1, c - 1] = 1
    return gun


# ---------------------------------------------------------
# Демонстрация
# ---------------------------------------------------------
def main() -> int:
    # Создание игры с классическими правилами (B3/S23)
    game = GameOfLife(size=(200, 200), wrap=True, rule_b="3678", rule_s="34678")

    # Случайное заполнение поля и добавление нескольких шаблонов
    game.randomize(p=0.4, seed=228)
    game.set_pattern((1, 1), glider())
    game.set_pattern((10, 2), glider())
    game.set_pattern((20, 20), gosper_glider_gun())

    # Запуск анимации (на 120 поколений, шаг 100 мс)
    ani = game.animate(frames=1000, interval=10, save=True, save_path="game_of_life.gif")
    return 0


if __name__ == "__main__":
    main()
