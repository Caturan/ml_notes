import numpy as np
import matplotlib.pyplot as plt

# Örnek veri
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.5, 3.0, 5.5, 5.0])

# Closed Form çözümü
def closed_form(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]  # X'e bir bias terimi ekleniyor
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best[1], theta_best[0]

# Closed Form ile en uygun m ve b değerlerini bulalım
m, b = closed_form(X, y)

# Sonuç çizgisini çiz
X_line = np.linspace(min(X), max(X), 100)
y_line = m * X_line + b

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(X_line, y_line, color='green', linestyle='-', label=f'Closed Form Solution (m={m:.2f}, b={b:.2f})')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Closed Form Solution for Linear Regression")
plt.show()

"""
en uygun m ve b değerlerini matematiksel olarak direkt çözerek elde eder ve en uygun regresyon doğrusunu çizer.
"""