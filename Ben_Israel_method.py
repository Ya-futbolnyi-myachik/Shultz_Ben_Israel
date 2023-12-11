import numpy as np
from numpy import linalg as LA


def gen_inv(original, tolerance=1e-7, max_iterations=1000, show_explanation=False):
    """
    Вычисляет псевдообратную матрицу (generalized inverse) итерационным методом Бен-Израэля.

    Параметры:
    - original (numpy.ndarray): Исходная матрица произволной размерности (m x n), где m - число строк, n - число столбцов.
    - tolerance (float, optional): Порог сходимости. Алгоритм завершит выполнение,
    когда sigma станет меньше tolerance. По умолчанию 1e-7.
    - max_iterations (int, optional): Максимальное количество итераций. По умолчанию 1000.
    - show_explanation (bool, optional): Флаг, если True, выводит информацию о каждой итерации. По умолчанию False.

    Возвращает:
    - numpy.ndarray or None: Псевдообратная матрица, если алгоритм сошелся, иначе None.

    Примечания:
    - Существуют матрицы полного строкового и полного столбцового рангов, алгоритм нахождения псевдообратной матрицы для них разный.
    - Если исходная матрица квадратная, то её псевдообратная и обратная матрицы равны.
    """

    frobenius_norm = LA.norm(original, ord='fro')   # Норма Фробениуса исходной матрицы
    alpha = 1.8 / frobenius_norm ** 2
    approx = alpha * original.transpose()           # Начальное приближение, с которого начинается цикл

    dimension = original.shape    # Размерность исходной матрицы, кортеж вида (m, n)
    m_or_n = min(dimension)       # Возвращает минимальное значение из (m, n)

    row_or_column = dimension.index(m_or_n)     # Флаг, принимает значения:
    # 0 (False) если число строк меньше числа столбцов (m < n)
    # 1 (True) если число столбцов меньше числа строк (m > n)

    unit = 2 * np.eye(m_or_n)    # Единичную матрицу размерности "m_or_n" умножаем на два
    sigma = 1                    # Задаем переменую-условие остновки цикла
    k = 0                        # Счетчик количества итераций

    while sigma >= tolerance and k < max_iterations:
        # Цикл срабатывает пока sigma не достигнет значения tolerance
        # или пока количество итераций не достигнет max_iterations

        k += 1
        approx_copy = np.copy(approx)   # Сохраняем предыдущее приближене для последующего вычисления коэф-та sigma

        if row_or_column and LA.matrix_rank(original) == m_or_n:
            # Если row_or_column == 1 (что соответствует True)
            # и ранг исходной матрицы равен n (числу столбцов),
            # то исходная матрица - матрица полного столбцового ранга

            a = approx.dot(original)    # Умножаем текущее приближение на исходную матрицу
            b = np.subtract(unit, a)    # Вычитаем из удвоенной единичной матриицы
            approx = b.dot(approx)      # Присваеваем новое значение текущему приближению

        elif not row_or_column and LA.matrix_rank(original) == m_or_n:
            # Если row_or_column == 0 (а not 0 = True)
            # и ранг исходной матрицы равен m (числу строк),
            # то исходная матрица - матрица полного строкового ранга

            a = original.dot(approx)    # Наобоорот, умножаем исходную матрицу на текущее приближение
            b = np.subtract(unit, a)    # Вычитаем из удвоенной единичной матриицы
            approx = approx.dot(b)      # Присваеваем новое значение текущему приближению

        else:
            if show_explanation:
                print("Исходная матрица не является матрицей полного ранга.")
            return None

        # Кубическая норма разности текущего и предыдущего приближений
        numerator = LA.norm(np.subtract(approx, approx_copy), np.inf)
        # Кубическая норма предыдущего приближения
        denominator = LA.norm(approx_copy, np.inf)
        # Коэф-т sigma есть отношение кубических норм, представленных выше
        sigma = numerator / denominator

        if show_explanation:
            # Выводит номер итерации, текущее значение sigma и текущее приближение
            print(f"Итерация №{k}\n\t\tСигма:{sigma}\n{approx}\n")

    if sigma <= tolerance:
        if show_explanation:
            print(f"Алгоритм сходится за {k} итераций.")
        return approx
    else:
        if show_explanation:
            print(f"Сходимость не достигнута после {max_iterations} итераций.")
        return None


# Пример работы алгоритма
original = np.array([[2, 1, 1, 3],
                     [1, 0, 0, -1],
                     [1, 1, 0, 4]])

gen_inv(original, show_explanation=True)







