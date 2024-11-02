import numpy as np
import matplotlib.pyplot as plt

# Örnek veri
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.5, 3.0, 5.5, 5.0])

# Brute Force çözümü
def brute_force(X, y, slope_range, intercept_range, step=0.1):
    min_error = float('inf')
    best_m, best_b = 0, 0
    for m in np.arange(slope_range[0], slope_range[1], step):
        for b in np.arange(intercept_range[0], intercept_range[1], step):
            y_pred = m * X + b
            error = np.mean((y - y_pred) ** 2)
            if error < min_error:
                min_error = error
                best_m, best_b = m, b
    return best_m, best_b

# Brute Force ile en uygun m ve b değerlerini bulalım
m, b = brute_force(X, y, slope_range=(-1, 2), intercept_range=(-1, 3))

# Sonuç çizgisini çiz
X_line = np.linspace(min(X), max(X), 100)
y_line = m * X_line + b

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(X_line, y_line, color='blue', linestyle='--', label=f'Brute Force Solution (m={m:.2f}, b={b:.2f})')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Brute Force Solution for Linear Regression")
plt.show()

"""
belirli bir aralikta m ve b değerlerini deneyerek en düşük hatayi bulur ve en uygun regresyon doğrusunu çizer.
"""