r1 = 0.14;
r2 = 0.1;  # riskless interest rate 1, 2
sig1 = 0.6;
sig2 = 0.3;  # volatility 1, 2
rho = 0.25;  # correlation
RiskFree=0.05
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
''''
plt.plot(s1)
plt.plot(s2)
plt.show()

'''
w1=(r1*sig2**2-r2*0.25*sig1*sig2)/(r1*sig2**2+r2*sig1**2-(r1+r2)*2*rho*sig1*sig2)
w2=1-w1
Ep=r1*w1+r2*w2
Sig=sp.sqrt(sig1**2*w1*w1+sig2**2*w2*w2+2*sig2*sig1*w1*w2*rho)
slpoe=(Ep-RiskFree)/Sig



#monte-carlo simulation
yy=[0]*1000
ssigg=[0]*1000
for i in range (1000):
  w11=random.random()
  w22=1-w11
  yy[i]=w11*r1+r2*w22
  ssigg[i]=sp.sqrt(w11*w11*sig1*sig1+w22*w22*sig2*sig2+2*w11*w22*rho*sig1*sig2)


a=min(ssigg)
b=max(ssigg)
c=np.linspace(a,b,500)
y=slpoe*c+0.05
plt.plot(c,y)
plt.plot(ssigg,yy)
plt.xlabel("volatility of portfolio")
plt.ylabel("expected return of portfolio")
plt.show()

