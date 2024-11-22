import numpy as np
from lagrange_interpolation import lagrange_interpolation


def find_optimal_n(func, n_values, num_samples=1001):
    optimal_n = None
    min_max_error = float('inf')

    for n in n_values:
        # Генерируем узлы интерполяции
        nodes = np.linspace(0, 2, n)
        y_nodes = func(nodes)

        x_fine = np.linspace(0, 2, num_samples)
        y_interpolated = lagrange_interpolation(x_fine, nodes, y_nodes)

        # Вычисляем максимальное отклонение от оригинальной функции
        max_error = np.max(np.abs(y_interpolated - func(x_fine)))

        # Проверяем, является ли текущее отклонение минимальным
        if max_error < min_max_error:
            min_max_error = max_error
            optimal_n = n

    return optimal_n, min_max_error
