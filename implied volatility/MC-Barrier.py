from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import random, ndarray, shape, arange, asarray
from math import *

plt.style.use('seaborn')


# MONTE CARLO SOLUTION
def mc(J, N, S0, T, r, sigma, Bar):
    dt = 1. / 365.
    DT = T / N
    L = int(T / dt)

    if T - L * dt > DT:
        L = L + 1

    X = ndarray(shape=(N + 1))
    P = ndarray(shape=(L + 1))
    for n in range(L + 1):
        P[n] = 0.0

    X_l = log(Bar / S0)

    for j in range(J):
        X[0] = 0.0
        Y = random.normal(loc=-.5 * sigma * sigma * DT, scale=sigma * sqrt(DT), size=N)
        for n in range(N):
            X[n + 1] = X[n] + Y[n]
            if X[n + 1] < X_l + r * (n + 1) * DT:
                tau = (n + 1) * DT
                for l in range(L + 1):
                    if l * dt < tau:
                        P[l] += 0.0
                    else:
                        P[l] += 1.0
                break

    return P


def an(S, sig, ba, t):
    mu_hat = -(sig ** 2 / 2)  # zero drift
    lam = mu_hat / sig ** 2
    a = (ba / S) ** (2 * lam)
    b = ((log(ba / S) + mu_hat * T) / (sig * sqrt(t)))
    c = ((log(ba / S) - mu_hat * T) / (sig * sqrt(t)))
    sol = a * norm.cdf(b) + norm.cdf(c)
    return sol


S0 = 1.5
Bar = 1.3
T = 1.0
r = 0.01
sigma = .20
N = 1000
dt = 1. / 365.
J = 10000

P = mc(J, N, S0, T, r, sigma, Bar)
print(shape(P))
for j in range(P.size):
    P[j] = P[j] / J
                                                                
print("\n")
print("%8s  %12s" % ("t", "P(tau < t )"))
for n in range(P.size):
    t = n * dt
    print("%8.4f  %12.6f" % (t, P[n]))

ap = an(S0, sigma, Bar, T)
print("\n") 
print("%8s  %12s" % ("t", "P(tau < t )"))
anl = []
for t in arange(dt, 1+2*dt, dt):
    ap = an(S0, sigma, Bar, t)
    anl.append(ap)
    print("%8.4f  %12.6f" % (t, ap))


X1 = asarray(anl)
C = range(P.size)
plt.plot(C, P, label='MC')
plt.plot(C, X1, label='Analytical')
plt.title('Monte Carlo/ Analytical CDF')
plt.xlabel('T')
plt.ylabel('Probability')
plt.legend()
plt.show()
