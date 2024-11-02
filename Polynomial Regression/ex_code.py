import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Örnek veri oluşturma
np.random.seed(0)
X = 2 * np.random.rand(100, 1) - 1  # -1 ve 1 arasında değerler
y = 1 + 0.5 * X - 0.2 * X**2 + np.random.randn(100, 1) * 0.1  # 2. dereceden polinom ve biraz gürültü

# Polinomsal özellik matrisi (X ve X^2 sütunlarını içerir)
X_poly = np.c_[np.ones((len(X), 1)), X, X**2]

# Gradient Descent için hiperparametreler
learning_rate = 0.1
n_iterations = 100
m = X_poly.shape[1]  # Parametre sayısı
theta = np.random.randn(m, 1)  # Rastgele başlangıç ağırlıkları

# Gradient Descent
theta_values = []
for iteration in range(n_iterations):
    gradients = 2 / len(X_poly) * X_poly.T @ (X_poly @ theta - y)
    theta -= learning_rate * gradients
    theta_values.append(theta.copy())

# Animasyon için figür ve eksen hazırlığı
fig, ax = plt.subplots()
ax.scatter(X, y, color='black', label='Data Points')
line, = ax.plot([], [], color='red', label='Polynomial Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression with Gradient Descent")

# Güncelleme fonksiyonu
def update(i):
    theta_i = theta_values[i]
    X_line = np.linspace(-1, 1, 100).reshape(100, 1)
    X_line_poly = np.c_[np.ones((100, 1)), X_line, X_line**2]
    y_line = X_line_poly @ theta_i
    line.set_data(X_line, y_line)
    return line,

# Animasyonu çalıştır
ani = FuncAnimation(fig, update, frames=n_iterations, repeat=False, blit=True)
plt.show()
 