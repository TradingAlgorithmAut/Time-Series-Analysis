import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Generate example with seasonal and trend components
np.random.seed(42)
n = 100
time = pd.date_range(start='2022-01-01', periods=n, freq='M') 
seasonal_component = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
trend_component = np.linspace(0, 20, num=n)
noise = np.random.normal(0, 1, size=n)
data = seasonal_component + trend_component + noise

# Perform STL decomposition
stl = STL(data,period=10,seasonal=13)
result = stl.fit()

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(time, data, label='Original Data')
plt.title('Original Time Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, result.trend, label='Trend Component', color='red')
plt.title('Trend Component')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, result.seasonal, label='Seasonal Component', color='green')
plt.title('Seasonal Component')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, result.resid, label='Residual Component', color='purple')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()
