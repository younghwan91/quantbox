import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.linalg import svd

def tls(X, Y):
    """
    Perform Total Least Squares regression.
    
    Parameters:
    X : numpy array
        Independent variable data matrix (m by n).
    Y : numpy array
        Dependent variable data matrix (m by p).
    
    Returns:
    B : numpy array
        TLS regression coefficients.
    """
    # Combine X and Y to form Z
    Z = np.hstack((X, Y))
    # Perform SVD
    U, S, Vt = svd(Z, full_matrices=False)
    
    # Extract the relevant blocks from V
    n = X.shape[1]
    V = Vt.T
    VXY = V[:n, n:]
    VYY = V[n:, n:]
    
    # Calculate TLS regression coefficients
    B = -np.dot(VXY, np.linalg.inv(VYY))
    
    return B

# Generate synthetic data for two assets
np.random.seed(42)
n = 100
X = np.random.rand(n, 1) * 10
Y = 2 * X + 1 + np.random.randn(n, 1) * 2  # Adding more noise to Y

# OLS Regression
ols = LinearRegression()
ols.fit(X, Y)
Y_ols = ols.predict(X)

# TLS Regression
B_tls = tls(X, Y)
Y_tls = X @ B_tls

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, label='Data', color='blue')
plt.plot(X, Y_ols, label='OLS Fit', color='red')
plt.plot(X, Y_tls, label='TLS Fit', color='green')
plt.xlabel('X (Asset 1)')
plt.ylabel('Y (Asset 2)')
plt.legend()
plt.title('OLS vs TLS for Pairs Trading')
plt.show()

# Compare gradients (hedge ratios)
ols_gradient = ols.coef_[0][0]
tls_gradient = B_tls[0][0]

print(f"OLS Hedge Ratio (Gradient): {ols_gradient}")
print(f"TLS Hedge Ratio (Gradient): {tls_gradient}")

# Calculate and compare the differences
difference = abs(ols_gradient - tls_gradient)
print(f"Difference in Hedge Ratios: {difference}")
