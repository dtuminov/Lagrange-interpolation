# Генерация формулы Лагранжа
def lagrange_formula(x_nodes, y_nodes):
    n = len(x_nodes)
    terms = []
    for i in range(n):
        term = f"{y_nodes[i]}"
        for j in range(n):
            if i != j:
                term += f" * (x - {x_nodes[j]}) / ({x_nodes[i]} - {x_nodes[j]})"
        terms.append(term)
    return " + ".join(terms)


