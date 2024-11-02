import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time 

# Örnek veri oluşturma
np.random.seed(0)
X = 2 * np.random.rand(15, 1) - 1  # -1 ile 1 arasında X değerleri
y = 1 + 0.5 * X - 0.2 * X**2 + np.random.randn(15, 1) * 0.1  # 2. dereceden polinom + gürültü

# Model derecelerini tanımla
degrees = [1, 3, 4, 5, 7, 8, 10, 13, 15, 20, 22]  # Model karmaşıklığını artırmak için farklı dereceler

# Animasyon için figür ve eksen hazırlığı
fig, ax = plt.subplots()
ax.scatter(X, y, color='black', label='Data Points')
line, = ax.plot([], [], color='red', label='Polynomial Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Underfitting to Overfitting in Polynomial Regression")

# Her derecede güncelleme fonksiyonu
def update(i):
    time.sleep(2)
    degree = degrees[i]
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Modelin çizimi için X değerlerini genişlet
    X_line = np.linspace(-1, 1, 100).reshape(100, 1)
    X_line_poly = poly_features.transform(X_line)
    y_line = model.predict(X_line_poly)
    
    line.set_data(X_line, y_line)
    ax.set_title(f"Polynomial Regression - Degree {degree}")
    return line,

# Animasyonu çalıştır
ani = FuncAnimation(fig, update, frames=len(degrees), repeat=False, blit=True)
plt.show()
