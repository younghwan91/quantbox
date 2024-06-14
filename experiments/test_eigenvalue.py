import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic prices using geometric Brownian motion
np.random.seed(42)
num_assets = 10
num_days = 500
initial_price = 100
mu = 0.0005
sigma = 0.01

dates = pd.date_range(start='2020-01-01', periods=num_days)
prices = pd.DataFrame(index=dates, columns=[f'Asset_{i+1}' for i in range(num_assets)])

for asset in prices.columns:
    prices[asset] = initial_price * np.exp(np.cumsum((mu - 0.5 * sigma**2) + sigma * np.random.randn(num_days)))

# Calculate log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Perform PCA
pca = PCA(n_components=1)
principal_components = pca.fit_transform(log_returns)

# Loadings (eigenvectors) corresponding to the principal component
loadings = pd.DataFrame(pca.components_.T, index=log_returns.columns, columns=['PC1'])

# Use the first principal component loadings as weights
weights = loadings['PC1']

# Calculate the synthetic spread (weighted combination of all assets using the eigenvector weights)
synthetic_spread = log_returns.dot(weights)

# Plot the synthetic spread
plt.figure(figsize=(12, 6))
plt.plot(synthetic_spread, label='Synthetic Spread')
plt.axhline(synthetic_spread.mean(), color='red', linestyle='--', label='Mean Spread')
plt.xlabel('Date')
plt.ylabel('Synthetic Spread')
plt.title('Synthetic Spread of Combined Assets')
plt.legend()
plt.show()

# Define entry and exit points based on the spread's deviation from the mean
entry_threshold = 2 * synthetic_spread.std()
exit_threshold = synthetic_spread.mean()

# Identify entry and exit points
entry_points = synthetic_spread[(synthetic_spread - synthetic_spread.mean()).abs() > entry_threshold]
exit_points = synthetic_spread[(synthetic_spread - synthetic_spread.mean()).abs() <= exit_threshold]

# Display entry and exit points
print("Entry points (indices):", entry_points.index)
print("Exit points (indices):", exit_points.index)
