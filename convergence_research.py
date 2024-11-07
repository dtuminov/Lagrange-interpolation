import numpy as np
from lagrange_interpolation import lagrange_interpolation


# Функция для вычисления максимального отклонения
def evaluate_max_deviation(func, n_start, n_end):
    deviations = []
    n_values = range(n_start, n_end + 1)
    x_fine = np.linspace(0, 2, 1001)  # Равномерная сетка из 1001 узла

    for n in n_values:
        # Настройка значений узлов интерполяции
        x_nodes = np.linspace(0, 2, n)
        y_nodes = func(x_nodes)

        # Вычисление значений интерполяционного многочлена
        y_interpolated = lagrange_interpolation(x_fine, x_nodes, y_nodes)

        # Вычисление максимального отклонения
        max_deviation = np.max(np.abs(y_interpolated - func(x_fine)))
        deviations.append(max_deviation)

    return n_values, deviations
