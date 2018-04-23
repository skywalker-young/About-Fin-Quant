import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

   # underlying price 1, 2
r1 = 0.07;
r2 = 0.03;  # riskless interest rate 1, 2
sig1 = 0.3;
sig2 = 0.6;  # volatility 1, 2
rho = 0.25;  # correlation

T=1
step=200
dt=T/step

s1=[0]*step
s2=[0]*step
s1[0]=10
s2[0]=20
dw,dw1,dw2=[0]*step,[0]*step,[0]*step
#S+ds
for i in range(step-1):
    dw[i + 1] = np.random.normal(0, 1) * sp.sqrt(dt)
  #s1[i+1]=sig1*s1[i]*np.random.normal(0,1)*sp.sqrt(dt)+r1*s1[i]*dt+s1[i]


for i in range(step-1):
    dw1[i+1]=np.random.normal(0,1)*sp.sqrt(dt)

for i in range(step):
    dw2[i]=rho*dw[i]+sp.sqrt(1-rho**2)*dw1[i]

for i in range (step-1):
    s1[i+1]=sig1*s1[i]*dw1[i]+r1*s1[i]*dt+s1[i]

for i in range (step-1):
    s2[i+1]=sig2*s2[i]*dw2[i]+r2*s2[i]*dt+s2[i]

plt.plot(s1)
plt.plot(s2)
plt.show()
