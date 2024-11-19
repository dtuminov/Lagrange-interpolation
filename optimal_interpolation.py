import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BarycentricInterpolator
from numpy.polynomial import Polynomial

# Определение исходной функции
def f(x):
    return np.abs(np.sin(4 * x)) * np.exp(2 * x)

# Определение первой функции
def f1(x):
    return (63 * x ** 5) / 8 - (35 * x ** 3) / 4 + (15 * x) / 8


# Метод наименьших квадратов
def least_squares_fit(x_nodes, y_nodes, degree):
    coeffs = np.polyfit(x_nodes, y_nodes, degree)
    poly = Polynomial(coeffs)
    return poly


# Интерполяционный многочлен в форме Ньютона
def newton_interpolation(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = np.zeros(n)
    coef[0] = y_nodes[0]

    def divided_difference(x, y):
        n = len(x)
        coeff = np.zeros((n, n))
        coeff[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                coeff[i][j] = (coeff[i + 1][j - 1] - coeff[i][j - 1]) / (x[i + j] - x[i])
        return coeff[0]

    div_diff = divided_difference(x_nodes, y_nodes)

    def newton_poly(x):
        p = div_diff[0]
        for i in range(1, n):
            term = div_diff[i]
            for j in range(i):
                term *= (x - x_nodes[j])
            p += term
        return p

    return newton_poly


# Интерполяция с помощью полиномов Лагранжа
def lagrange_interpolation(x_nodes, y_nodes):
    def lagrange(x):
        total = 0
        for i in range(len(x_nodes)):
            term = y_nodes[i]
            for j in range(len(x_nodes)):
                if j != i:
                    term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
            total += term
        return total

    return lagrange


# Основная функция для вычисления максимальных отклонений
def max_deviation(func, interp_func, x_fine):
    y_exact = func(x_fine)
    y_interp = interp_func(x_fine)
    return np.max(np.abs(y_exact - y_interp))

def evaluate_max_deviation(func, n_start, n_end):
    deviations = []
    n_values = range(n_start, n_end + 1)
    x_fine = np.linspace(0, 2, 1001)  # Равномерная сетка из 1001 узла

    for n in n_values:
        # Настройка значений узлов интерполяции
        x_nodes = np.linspace(0, 2, n)
        y_nodes = func(x_nodes)

        # Создание кубического сплайна
        cs = CubicSpline(x_nodes, y_nodes)

        # Вычисление значений интерполяционного многочлена
        y_interpolated = cs(x_fine)

        # Вычисление максимального отклонения
        max_deviation = np.max(np.abs(y_interpolated - func(x_fine)))
        deviations.append(max_deviation)

    return n_values, deviations

# Настройка узлов интерполяции
n_nodes = [3, 5, 9, 17]  # Различные количества узлов
x_fine = np.linspace(0, 2, 1000)  # Тонкая выборка для построения графиков

# Словарь для хранения максимальных отклонений
deviations = {}

# Первый набор данных
print("Первый набор данных")
# Различные методы интерполяции
for n in n_nodes:
    x_nodes = np.linspace(0, 2, n)
    y_nodes = f1(x_nodes)

    # Различные методы интерполяции
    ls_poly = least_squares_fit(x_nodes, y_nodes, degree=4)
    spline_interp = CubicSpline(x_nodes, y_nodes)
    newton_interp = newton_interpolation(x_nodes, y_nodes)
    lagrange_interp = lagrange_interpolation(x_nodes, y_nodes)

    # Вычисление отклонений
    deviations[f'Least Squares (n={n})'] = max_deviation(f1, ls_poly, x_fine)
    deviations[f'Spline (n={n})'] = max_deviation(f1, spline_interp, x_fine)
    deviations[f'Lagrange (n={n})'] = max_deviation(f1, lagrange_interp, x_fine)

# Вывод отклонений в консоль
for method, deviation in deviations.items():
    print(f'{method}: {deviation}')

print(f"Минимальное максимальное отклонение {min(deviations, key=deviations.get)}: {min(deviations.values())}")

# Второй набор данных
print("Второй набор данных")
for n in n_nodes:
    x_nodes = np.linspace(0, 2, n)
    y_nodes = f(x_nodes)

    # Различные методы интерполяции
    ls_poly = least_squares_fit(x_nodes, y_nodes, degree=4)
    spline_interp = CubicSpline(x_nodes, y_nodes)
    newton_interp = newton_interpolation(x_nodes, y_nodes)
    lagrange_interp = lagrange_interpolation(x_nodes, y_nodes)

    # Вычисление отклонений
    deviations[f'Least Squares (n={n})'] = max_deviation(f, ls_poly, x_fine)
    deviations[f'Spline (n={n})'] = max_deviation(f, spline_interp, x_fine)
    deviations[f'Lagrange (n={n})'] = max_deviation(f, lagrange_interp, x_fine)

# Вывод отклонений в консоль
for method, deviation in deviations.items():
    print(f'{method}: {deviation}')

print(f"Минимальное максимальное отклонение {min(deviations, key=deviations.get)}: {min(deviations.values())}")

# Построение графиков
plt.figure(figsize=(12, 8))
plt.plot(x_fine, f(x_fine), label='Исходная функция f(x)', color='black', linewidth=2)

for n in n_nodes:
    x_nodes = np.linspace(0, 2, n)
    y_nodes = f(x_nodes)
    # Построение интерполяций
    plt.plot(x_fine, lagrange_interpolation(x_nodes, y_nodes)(x_fine), label=f'Lagrange (n={n})', linestyle='--')
    plt.plot(x_fine, CubicSpline(x_nodes, y_nodes)(x_fine), label=f'Spline (n={n})', linestyle=':')
    plt.plot(x_fine, least_squares_fit(x_nodes, y_nodes, degree= 4)(x_fine), label=f'LSM (n={n})', linestyle='-.')

plt.title('Интерполяция функции')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

# Построение графиков
plt.figure(figsize=(12, 8))
plt.plot(x_fine, f1(x_fine), label='Исходная функция f(x)', color='black', linewidth=2)

for n in n_nodes:
    x_nodes = np.linspace(0, 2, n)
    y_nodes = f1(x_nodes)
    # Построение интерполяций
    plt.plot(x_fine, lagrange_interpolation(x_nodes, y_nodes)(x_fine), label=f'Lagrange (n={n})', linestyle='--')
    plt.plot(x_fine, CubicSpline(x_nodes, y_nodes)(x_fine), label=f'Spline (n={n})', linestyle=':')
    plt.plot(x_fine, least_squares_fit(x_nodes, y_nodes, degree= 4)(x_fine), label=f'LSM (n={n})', linestyle='-.')

plt.title('Интерполяция функции')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

n1, deviations_f1 = evaluate_max_deviation(f1, 2, 17)
n2, deviations_f2 = evaluate_max_deviation(f, 2, 17)

# Визуализация зависимости отклонения от числа узлов
plt.figure(figsize=(14, 6))

# График для первой функции
plt.subplot(1, 2, 1)
plt.plot(n1, deviations_f1, marker='o', label='f1(x)', color='blue')
plt.title('Максимальное отклонение для f1(x)')
plt.xlabel('Число узлов (n)')
plt.ylabel('Максимальное отклонение')
plt.grid()
plt.xticks(n1)

# График для второй функции
plt.subplot(1, 2, 2)
plt.plot(n2, deviations_f2, marker='o', label='f2(x)', color='orange')
plt.title('Максимальное отклонение для f2(x)')
plt.xlabel('Число узлов (n)')
plt.ylabel('Максимальное отклонение')
plt.grid()
plt.xticks(n2)

plt.tight_layout()
plt.show()