import numpy as np
from matplotlib import pyplot as plt
from math import *
import numpy.random as nrand

plt.style.use('seaborn')

# Parameters

r0 = 0.02  # Starting interest rate
time = 100  # Simulation time
delta = 1 / 252  # Delta time
sigma = 0.03  # Volatility
ou_a = 20  # Rate of mean reversion
ou_mu = 0.02  # Long run average
end = 90  # Time = T
end = end + 1
i = 1000  # Number of simulations
barrier = 0.01  # Level of barrier


def brownian_motion_log_returns(dt, sig):
    sqrt_delta_sigma = sqrt(dt) * sig
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=time)


def ornstein_uhlenbeck(t, a, mu, dt):
    paths = [r0]
    brownian_motion_returns = brownian_motion_log_returns(delta, sigma)
    for i in range(1, t):
        drift = a * (mu - paths[i - 1]) * dt
        randomness = brownian_motion_returns[i - 1]
        paths.append(paths[i - 1] + drift + randomness)
    return paths


# MONTE CARLO SIMULATION

paths = []
for i in range(i):
    level = ornstein_uhlenbeck(time, ou_a, ou_mu, delta)
    paths.append(level)

paths = np.asarray(paths)
paths = paths.T
new_paths = np.delete(paths, np.s_[end:], 0)  # Remove paths beyond T
print(np.shape(new_paths))

# MC projection graph

plt.plot(new_paths, lw=0.5)
plt.axhline(y=barrier, color='b', linestyle='-', lw=0.7)
plt.title('Monte Carlo Simulations')
plt.xlabel('Time')
plt.ylabel('Level')
plt.show()


def prob(path, b, i):
    count = 0
    for col in path.T:
        a = min(col)
        if a < b:
            count = count + 1
    p = count / i
    return p


mcp = prob(new_paths, barrier, i)
print('Monte Carlo probability of touching:', mcp)
