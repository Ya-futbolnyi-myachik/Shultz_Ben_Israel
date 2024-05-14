import numpy as np
from numpy import linalg as LA


def shultz_inv(original, tolerance=1e-7, max_iterations=1000, show_explanation=False):
    """
    Вычисляет обратную матрицу итерационным методом Шульца.

    Параметры:
    - original (numpy.ndarray): Исходная квадратная невыроденная матрица.
    - tolerance (float): Условие остановки цикла. Алгоритм завершит выполнение,
    когда sigma станет меньше tolerance. По умолчанию 1e-7.
    - max_iterations (int): Максимальное количество итераций. По умолчанию 1000.
    - show_explanation (bool): Флаг, если True, выводит информацию о каждой итерации. По умолчанию False.

    Возвращает:
    - numpy.ndarray or None: Обратная матрица, если алгоритм сошелся, иначе None.

    Примечания:
    - Если исходная матрица вырожденная, функция вернет None.
    - Если исходная матрица не квадратная, функция вернет None.
    - Если алгоритм сошелся, будет выведено сообщение о количестве итераций.

    """
    try:
        det = LA.det(original)
        if det == 0:
            if show_explanation:
                print("Исходная матрица является вырожденной. Невозможно вычислить обратную матрицу.")
            return None
    except np.linalg.LinAlgError:
        if show_explanation:
            print("Исходная матрица не квадратная")
        return None

    frobenius_norm = LA.norm(original, ord='fro')   # Норма Фробениуса исходной матрицы
    alpha = 1.8 / frobenius_norm ** 2
    approx = alpha * original.transpose()           # Начальное приближение, с которого начинается цикл

    if show_explanation:
        print(f'Норма Фробениуса:\n{frobenius_norm}\n\n'
              f'Альфа равно:\n{alpha}\n\n'
              f'Транспонированная матрица:\n{original.transpose()}\n\n'
              f'Начальное приближение:\n{approx}\n')

    n = original.shape[0]   # Размерность исходной матрицы
    unit = 2 * np.eye(n)    # Единичную матрицу той же размерности умножаем на два
    sigma = 1               # Задаем переменую-условие остновки цикла
    k = 0                   # Счетчик количества итераций

    while sigma >= tolerance and k < max_iterations:
        # Цикл срабатывает пока sigma не достигнет значения tolerance
        # или пока количество итераций не достигнет max_iterations

        k += 1
        a = original.dot(approx)        # Умножаем исходную матрицу на текущее приближение
        b = np.subtract(unit, a)        # Вычитаем из удвоенной единичной матриицы
        approx_copy = np.copy(approx)   # Сохраняем предыдущее приближене для последующего вычисления коэф-та sigma
        approx = approx.dot(b)          # Присваеваем новое значение текущему приближению

        # Кубическая норма разности текущего и предыдущего приближений
        numerator = LA.norm(np.subtract(approx, approx_copy), np.inf)
        # Кубическая норма предыдущего приближения
        denominator = LA.norm(approx_copy, np.inf)
        # Коэф-т sigma есть отношение кубических норм, представленных выше
        sigma = numerator / denominator

        if show_explanation:
            # Выводит номер итерации, текущее значение sigma и текущее приближение
            print(f"Итерация №{k}\n\t\tСигма:{sigma}\n{approx}\n")

    if sigma < tolerance:
        if show_explanation:
            print(f"Алгоритм сходится за {k} итераций.")
        return approx
    else:
        if show_explanation:
            print(f"Сходимость не достигнута после {max_iterations} итераций.")
        return None


# Пример работы алгоритма
original = np.array([[1, 2, 1],
                     [0, 1, 0],
                     [0, 2, 2]])
shultz_inv(original, show_explanation=True)
