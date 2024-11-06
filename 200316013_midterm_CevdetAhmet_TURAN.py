# Cevdet Ahmet TURAN    
# 200316013

import pandas as pd
import matplotlib.pyplot as plt
import locale

locale.setlocale(locale.LC_TIME, 'tr_TR.UTF-8')  


df = pd.read_excel('GramAltin_5yillikVeri_241106.xlsx')


print(df.head())

df['Date'] = pd.to_datetime(df['Tarih'], format='%d %B %Y', errors='coerce')

print(df['Date'].isna().sum())

df = df.dropna(subset=['Date', 'Fiyat'])

df['Fiyat'] = df['Fiyat'].astype(str).str.replace(',', '.').astype(float)

df['Days'] = (df['Date'] - df['Date'].min()).dt.days

print(df.head())

x = df['Days'].values
y = df['Fiyat'].values

n = len(x)
sum_x = x.sum()
sum_y = y.sum()
sum_xy = (x * y).sum()
sum_x2 = (x**2).sum()

beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
beta_0 = (sum_y - beta_1 * sum_x) / n

print(f"Slope (beta_1): {beta_1}")
print(f"Intercept (beta_0): {beta_0}")

y_pred = beta_0 + beta_1 * x

specific_day = 5000 
predicted_price = beta_0 + beta_1 * specific_day
print(f"Predicted price for day {specific_day}: {predicted_price}")

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('Days')
plt.ylabel('Gold Price')
plt.title('Gold Price Prediction using Simple Linear Regression')
plt.legend()
plt.show()

# ------------------------------

# TASK2 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('GramAltin_5yillikVeri_241106.xlsx')

df['Date'] = pd.to_datetime(df['Tarih'], format='%d %B %Y', errors='coerce')

df = df.dropna(subset=['Date', 'Fiyat'])

df['Fiyat'] = df['Fiyat'].astype(str).str.replace(',', '.').astype(float)

df['Days'] = (df['Date'] - df['Date'].min()).dt.days

df['Days'] = pd.to_numeric(df['Days'], errors='coerce')

x = df['Days'].values
y = df['Fiyat'].values

def polynomial_regression(degree, alpha=1e-5):
    X_poly = np.vstack([x**i for i in range(1, degree+1)]).T
    
    X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]  
    
    XTX = X_poly.T @ X_poly
    XTX += alpha * np.eye(X_poly.shape[1])  
    
    beta = np.linalg.inv(XTX) @ X_poly.T @ y  
    
    return X_poly, beta

plt.figure(figsize=(15, 10))
for degree in range(2, 9):
    X_poly, beta = polynomial_regression(degree)
    
    y_pred = X_poly @ beta
    
    sort_index = np.argsort(x)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]
    y_pred_sorted = y_pred[sort_index]
    
    plt.subplot(3, 3, degree-1) 
    plt.scatter(x, y, color='blue', label='Actual data')
    plt.plot(x_sorted, y_pred_sorted, color='red', label=f'Poly Degree {degree}')
    plt.xlabel('Days')
    plt.ylabel('Gold Price')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()

plt.tight_layout()
plt.show()
"""