import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Örnek veri oluşturma
np.random.seed(0)
X = 2 * np.random.rand(100, 1) - 1  # -1 ile 1 arasında X değerleri
y = 1 + 0.5 * X - 0.2 * X**2 + np.random.randn(100, 1) * 0.1  # 2. dereceden polinom + gürültü

# Mini-Batch Gradient Descent parametreleri
learning_rate = 0.005
n_iterations = 1000
batch_size = 16

# Mini-Batch Gradient Descent fonksiyonu
def mini_batch_gradient_descent(X, y, theta, learning_rate, n_iterations, batch_size):
    m = len(X)
    rss_progress = []
    for iteration in range(n_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            gradients = 2/batch_size * X_batch.T @ (X_batch @ theta - y_batch)
            theta -= learning_rate * gradients
        
        y_pred = X @ theta
        rss = np.mean((y - y_pred) ** 2)
        rss_progress.append(rss)
        
    return theta, rss_progress

# Polinom derecelerini belirle
degrees = [1, 3, 5, 7, 11, 15, 19]

# RSS değerlerini ayrı bir grafikte çizmek için hazırlık
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    np.random.seed(42)
    theta = np.random.randn(X_poly.shape[1], 1)
    
    theta_best, rss_progress = mini_batch_gradient_descent(X_poly, y, theta, learning_rate, n_iterations, batch_size)
    
    plt.plot(rss_progress, label=f'Degree {degree}')

plt.xlabel("Iteration")
plt.ylabel("RSS")
plt.title("RSS Progress for Different Degrees")
plt.legend()

# Polinomsal regresyon çizgilerini çiz
plt.subplot(1, 2, 2)
X_line = np.linspace(-1, 1, 100).reshape(100, 1)
plt.scatter(X, y, color='black', label='Data Points')

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    X_line_poly = poly_features.transform(X_line)
    
    np.random.seed(42)
    theta = np.random.randn(X_poly.shape[1], 1)
    theta_best, _ = mini_batch_gradient_descent(X_poly, y, theta, learning_rate, n_iterations, batch_size)
    
    y_line = X_line_poly @ theta_best
    plt.plot(X_line, y_line, label=f'Degree {degree}')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression Lines for Different Degrees")
plt.legend()

plt.tight_layout()
plt.show()
