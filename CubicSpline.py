import numpy as np

class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x) - 1
        self.a = self.y.copy()

        # Параметры сплайна
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n + 1)
        self.d = np.zeros(self.n)

        # Решение системы уравнений
        self._compute_coefficients()

    def _gauss_elimination(self, A, b):
        n = len(b)
        # Прямой ход
        for i in range(n):
            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
                b[j] -= factor * b[i]
        
        # Обратный ход
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        return x

    def _compute_coefficients(self):
        h = np.diff(self.x)
        A = np.zeros((self.n + 1, self.n + 1))
        b = np.zeros(self.n + 1)

        # Условия для первого и последнего узлов
        A[0, 0] = 1
        A[self.n, self.n] = 1

        for i in range(1, self.n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3 * ((self.a[i + 1] - self.a[i]) / h[i] - (self.a[i] - self.a[i - 1]) / h[i - 1])

        # Решение системы A * c = b
        self.c = self._gauss_elimination(A, b)

        for i in range(self.n):
            self.b[i] = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (2 * self.c[i] + self.c[i + 1]) / 3
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * h[i])

    def evaluate(self, x_eval):
        x_eval = np.array(x_eval)
        y_eval = np.zeros_like(x_eval)

        for i in range(self.n):
            mask = (x_eval >= self.x[i]) & (x_eval <= self.x[i + 1])
            if np.any(mask):
                dx = x_eval[mask] - self.x[i]
                y_eval[mask] = (self.a[i] +
                                self.b[i] * dx +
                                self.c[i] * dx ** 2 +
                                self.d[i] * dx ** 3)

        return y_eval
