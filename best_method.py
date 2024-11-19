import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Определим целевую функцию
def f(x):
    return np.abs(np.sin(4 * x)) * np.exp(2 * x)

# Определим диапазон и узлы от 0 до 2
x = np.linspace(0, 2, 20)
y = f(x)

# Подготовим более плотный диапазон для отображения
x_dense = np.linspace(0, 2, 100)

# Интерполяция методом линейной интерполяции
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_dense)

# Интерполяция методом кубических сплайнов
cubic_interp = interp1d(x, y, kind='cubic')
y_cubic = cubic_interp(x_dense)

# Метод наименьших квадратов с полиномами (2-й степени)
def polynomial_model(x, a, b, c):
    return a * x**2 + b * x + c

# Подсчёт коэффициентов полинома
popt, _ = curve_fit(polynomial_model, x, y)

# Получаем значения полинома
y_poly = polynomial_model(x_dense, *popt)

# Функция для вычисления ошибки
def compute_error(true_values, approx_values):
    return np.max(np.abs(true_values - approx_values))

# Вычисление ошибок
error_linear = compute_error(f(x_dense), y_linear)
error_cubic = compute_error(f(x_dense), y_cubic)
error_poly = compute_error(f(x_dense), y_poly)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_dense, f(x_dense), label='f(x)', color='black', linewidth=2)
plt.plot(x_dense, y_linear, label='Линейная интерполяция', linestyle='--')
plt.plot(x_dense, y_cubic, label='Кубическая интерполяция', linestyle='--')
plt.plot(x_dense, y_poly, label='Полином 2-й степени', linestyle='--')
plt.scatter(x, y, color='red', zorder=5, label='Измеренные узлы')

# Настройка графика
plt.title('Приближение функции f(x) на интервале [0, 2]')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

# Вывод ошибок
print(f'Ошибка линейной интерполяции: {error_linear}')
print(f'Ошибка кубической интерполяции: {error_cubic}')
print(f'Ошибка полинома 2-й степени: {error_poly}')