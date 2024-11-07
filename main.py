import numpy as np
import matplotlib.pyplot as plt
from formula import lagrange_formula
from lagrange_interpolation import lagrange_interpolation
from find_optimal import find_optimal_n
from convergence_research import evaluate_max_deviation


# Определение первой функции
def f1(x):
    return (63 * x ** 5) / 8 - (35 * x ** 3) / 4 + (15 * x) / 8


# Определение второй функции
def f2(x):
    return np.abs(np.sin(4 * x)) * np.exp(2 * x)


n_values = [3, 5, 9, 17]

# Настройка значений узлов интерполяции для первой функции
n1 = 5  # Число узлов для первой функции
x_nodes1 = np.linspace(0, 2, n1)
y_nodes1 = f1(x_nodes1)

# Настройка значений узлов интерполяции для второй функции
n2 = 9  # Число узлов для второй функции
x_nodes2 = np.linspace(0, 2, n2)
y_nodes2 = f2(x_nodes2)

# Создание значений для оси x для интерполяции первой функции
x_values1 = np.linspace(0, 2, 1001)  # Новый диапазон x для первой функции
y_values1 = lagrange_interpolation(x_values1, x_nodes1, y_nodes1)

# Создание значений для оси x для интерполяции второй функции
x_values2 = np.linspace(0, 2, 1001)  # Новый диапазон x для второй функции
y_values2 = lagrange_interpolation(x_values2, x_nodes2, y_nodes2)

# Равномерная сетка из 1001 узла для исследуемых функций
x_fine1 = np.linspace(0, 2, 1001)
x_fine2 = np.linspace(0, 2, 1001)

# Вычисление значений интерполяционных многочленов
y_interpolated1 = lagrange_interpolation(x_fine1, x_nodes1, y_nodes1)
y_interpolated2 = lagrange_interpolation(x_fine2, x_nodes2, y_nodes2)

# Вычисление максимальных отклонений
max_deviation1 = np.max(np.abs(y_interpolated1 - f1(x_fine1)))
max_deviation2 = np.max(np.abs(y_interpolated2 - f2(x_fine2)))

# Вывод максимальных отклонений
print(f'Максимальное отклонение |Pn(x) - f1(x)| = {max_deviation1}')
print('Формула Лагранжа для f1(x):')
print(lagrange_formula(x_nodes1, y_nodes2))

print(f'Максимальное отклонение |Pn(x) - f2(x)| = {max_deviation2}')
print('Формула Лагранжа для f2(x):')
print(lagrange_formula(x_nodes2, y_nodes2))

# Нахождение оптимального n для каждой функции (n = 3, 5, 9, 17)
optimal_n1, min_deviation1 = find_optimal_n(f1, n_values, 1001)
optimal_n2, min_deviation2 = find_optimal_n(f2, n_values, 1001)

# Вывод оптимальных n и соответствующих максимальных отклонений
print(f'Оптимальное количество узлов для f1(x): {optimal_n1}, минимальное максимальное отклонение: {min_deviation1}')
print(f'Оптимальное количество узлов для f2(x): {optimal_n2}, минимальное максимальное отклонение: {min_deviation2}')

# Исследование интерполяции для n в [1,17]
n1, deviations_f1 = evaluate_max_deviation(f1, 1, 17)
n2, deviations_f2 = evaluate_max_deviation(f2, 1, 17)

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

# Визуализация
plt.figure(figsize=(12, 8))

# График первой функции и её интерполяции
plt.subplot(2, 1, 1)
plt.plot(x_values1, y_values1, label='Интерполяционный многочлен Лагранжа f1(x)', color='blue')
plt.scatter(x_nodes1, y_nodes1, color='red', label='Узлы интерполяции', zorder=5)
plt.plot(x_values1, f1(x_values1), label='Исходная функция f1(x)', color='green', linestyle='dashed')
plt.title('Интерполяция для f1(x) = (63 * x^5) / 8 - (35 * x^3) / 4 + (15 * x) / 8')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.xlim(0, 2)
plt.ylim(-40, 220)  # Ограничение оси y для первой функции

# График второй функции и её интерполяции
plt.subplot(2, 1, 2)
plt.plot(x_values2, y_values2, label='Интерполяционный многочлен Лагранжа f2(x)', color='blue')
plt.scatter(x_nodes2, y_nodes2, color='red', label='Узлы интерполяции', zorder=5)
plt.plot(x_values2, f2(x_values2), label='Исходная функция f2(x)', color='green', linestyle='dashed')
plt.title('Интерполяция для f2(x) = |sin(4x)| * exp(2x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.xlim(0, 2)
plt.ylim(-30, 70)  # Ограничение оси y для второй функции

plt.tight_layout()
plt.show()