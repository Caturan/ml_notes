
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_excel('GramAltin_5yillikVeri_241106.xlsx')

# Parse the 'Date' column manually with the correct format
df['Date'] = pd.to_datetime(df['Tarih'], format='%d %B %Y', errors='coerce')

# Handle missing values in 'Date' and 'Fiyat' columns
df = df.dropna(subset=['Date', 'Fiyat'])

# Replace commas with periods in 'Fiyat' and convert to numeric
df['Fiyat'] = df['Fiyat'].astype(str).str.replace(',', '.').astype(float)

# Ensure that 'Days' is the difference in days from the first date
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Ensure 'Days' is numeric
df['Days'] = pd.to_numeric(df['Days'], errors='coerce')

# Extract the features (Days) and target (Gold Price)
x = df['Days'].values
y = df['Fiyat'].values

# Function to perform polynomial regression with ridge regularization
def polynomial_regression(degree, alpha=1e-5):
    # Generate polynomial features (x^1, x^2, ..., x^degree)
    X_poly = np.vstack([x**i for i in range(1, degree+1)]).T
    
    # Add a column of ones for the intercept
    X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]  # Adding intercept
    
    # Regularization: Add a small value (alpha) to the diagonal of X^T X
    XTX = X_poly.T @ X_poly
    XTX += alpha * np.eye(X_poly.shape[1])  # Add alpha to diagonal for regularization
    
    # Calculate the coefficients using the regularized normal equation
    beta = np.linalg.inv(XTX) @ X_poly.T @ y  # Normal equation with regularization
    
    return X_poly, beta

# Plotting the polynomial regression for degrees 2 to 8
plt.figure(figsize=(15, 10))
for degree in range(2, 9):
    # Perform polynomial regression for the current degree
    X_poly, beta = polynomial_regression(degree)
    
    # Calculate the predictions
    y_pred = X_poly @ beta
    
    # Sort the data for plotting (to get a smooth curve)
    sort_index = np.argsort(x)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]
    y_pred_sorted = y_pred[sort_index]
    
    # Plot the actual data and the regression line
    plt.subplot(3, 3, degree-1)  # 3x3 grid of subplots
    plt.scatter(x, y, color='blue', label='Actual data')
    plt.plot(x_sorted, y_pred_sorted, color='red', label=f'Poly Degree {degree}')
    plt.xlabel('Days')
    plt.ylabel('Gold Price')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()

plt.tight_layout()
plt.show()
