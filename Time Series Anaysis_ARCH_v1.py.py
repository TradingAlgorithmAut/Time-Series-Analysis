import numpy as np
import matplotlib.pyplot as plt

#Set seed
np.random.seed(42)

#Parameters
n = 1000  # Number steps
alpha = [0.1, 0.2, 0.3]  # Coefficients
c = 0.05  # Constant

#white noise
epsilon = np.random.normal(0, 1, size=n)

#Initialize array
sigma_sq = np.zeros(n)

#ARCH(3)
for t in range(3, n):
    sigma_sq[t] = c + alpha[0] * sigma_sq[t-1]**2 + alpha[1] * sigma_sq[t-2]**2 + alpha[2] * sigma_sq[t-3]**2 + 0.1 * epsilon[t-1]**2

#Plot
plt.figure(figsize=(10, 5))
plt.plot(sigma_sq, label='ARCH(3)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()
