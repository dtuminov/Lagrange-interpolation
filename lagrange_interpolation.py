def lagrange_interpolation(x, x_nodes, y_nodes):
    L = 0
    n = len(x_nodes)
    for i in range(n):
        # Вычисление базисного полинома L_i
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        L += y_nodes[i] * L_i
    return L
