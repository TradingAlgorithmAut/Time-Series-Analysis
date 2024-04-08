import numpy as np
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

omega = 0.01
alpha = np.array([0.1, 0.2])
beta = np.array([0.3, 0.4, 0.2])

#white noise
n = 1000
epsilons = np.random.normal(0, 1, size=n)

sigma_sq = np.zeros(n)
returns = np.zeros(n)

for t in range(3, n):
    sigma_sq[t] = omega + alpha[0] * epsilons[t-1]**2 + alpha[1] * epsilons[t-2]**2 \
                  + beta[0] * sigma_sq[t-1] + beta[1] * sigma_sq[t-2] + beta[2] * sigma_sq[t-3]
    returns[t] = np.random.normal(0, np.sqrt(sigma_sq[t]))

# Plot simulated returns
plt.figure(figsize=(10, 5))
plt.plot(returns, label='Simulated Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()

