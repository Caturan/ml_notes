import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Örnek veri
X = np.array([9, 3, 2, 7, 11])
y = np.array([15, 5, 30, 5, 9])

# Gradient Descent ile animasyon için gerekli adımlar
learning_rate = 0.01
n_iterations = 50
m, b = 0, 0  # Başlangıç değerleri
N = len(X)
m_values, b_values = [], []  # Gradient Descent sürecini takip etmek için

# Gradient Descent döngüsü
for _ in range(n_iterations):
    y_pred = m * X + b
    gradient_m = -(2/N) * np.sum(X * (y - y_pred))
    gradient_b = -(2/N) * np.sum(y - y_pred)
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b
    m_values.append(m)
    b_values.append(b)

# Animasyon fonksiyonu
fig, ax = plt.subplots()
ax.scatter(X, y, color='black', label='Data Points')
line, = ax.plot([], [], color='red', label='Gradient Descent Line')
connection_line, = ax.plot(X, y, color='blue', linestyle='--', label='Point Connections')  # Veri noktaları arasındaki bağlantı

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Gradient Descent Progression with Point Connections")

def update(i):
    # Her adımda yeni bir regresyon doğrusu çizer
    y_line = m_values[i] * X + b_values[i]
    line.set_data(X, y_line)
    print(y_line)
    return line, connection_line  # Veri noktaları arasındaki bağlantıyı da döndürüyoruz

# Animasyonu çalıştır
ani = FuncAnimation(fig, update, frames=n_iterations, repeat=False, blit=True)
plt.show()
