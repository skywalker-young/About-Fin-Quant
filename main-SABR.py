import matplotlib.pyplot as plt
from pylab import cm
from scipy.optimize import curve_fit
from scipy import interpolate
import SABR as sa
from mpl_toolkits.mplot3d import Axes3D

def time_tango(dates):
    return datetime.strptime("{}".format(dates), "%Y-%m-%d")



convDates = 365.0; # date convention
V0 = 2952.48; # EUROSTOXX50 index at '2016-01-15'
qd = '2016-01-15'; # quote date
cc = 0.015; # example cost of carry
xl=pd.read_excel('C:/Users/User/Desktop/TC/20160101_EUROSTOXX50.xlsx')
euro=xl
'''
euro=xl.head()
print(euro['SS'])
'''
euro['STRIKE'] *= 1.0
euro['IMVOL'] = 0.0  #
euro['TTM'] = 0.0    #所有列数变为0
expirySet = pd.Series(euro['EXP_DATE'].values.ravel()).unique()
stkSet = pd.Series(euro['STRIKE'].values.ravel()).unique()
aa=stkSet[48:57]
'''以上把表格里所有单一值提取了'''
sc = euro.loc[(euro['QUOTE_DATE'] == qd)&(euro['OPTION_TYPE'] == 'Call')]

sp = euro.loc[(euro['QUOTE_DATE'] == qd)&(euro['OPTION_TYPE'] == 'Put')]

# init

ttm = np.zeros(len(expirySet))
vol=[[0 for i in range(9)]for i in range(9)]
vol=np.array(vol).astype(np.float32)
a=[0]*9
b=[0]*9
for it in range(len(expirySet)):
    expiry = expirySet[it]
    ttm[it] = 1.0 * (time_tango(expiry) - time_tango(qd)).days / convDates
    '''qd今天引用的日子，ttm time to maturity'''
    # select time slice
    sc1 = sc.loc[sc['EXP_DATE'] == expiry]   # call的九个不同到期日
    sp1 = sp.loc[sp['EXP_DATE']==expiry  ]
#print(ttm) #[0.09589041 0.17260274 0.42191781 0.67123288 0.92054795 1.41917808 1.91780822 2.41643836 2.91232877]
    stkcp = list(set(sc1['STRIKE']) & set(sp1['STRIKE']))
    tmpcp = np.zeros((len(stkcp), 3))
    #print(len(stkcp))#70 73 80 59 56 54 54 59 59 49
    for i in range(len(stkcp)):
        tmpcp[i, 0] = stkcp[i]
        tmpcp[i, 1] = sc1[sc1['STRIKE'] == stkcp[i]]['PRICE']
        tmpcp[i, 2] = sp1[sp1['STRIKE'] == stkcp[i]]['PRICE']
        #print(i)#0-69 0-72...0-48

    dif = abs(tmpcp[:, 1] - tmpcp[:, 2])
    # minimum ((near the) At The Money)
    mindif = np.min(dif)
    minK = tmpcp[dif == mindif, 0]
    minKIdx = np.where(tmpcp[:, 0] == minK)[0][0]
    # synthetic forward(S = C-P + Ke^(-cc*T)) by using Put-Call parity
    mV0 = tmpcp[minKIdx, 1] - tmpcp[minKIdx, 2] + tmpcp[minKIdx, 0] * exp(-(cc)*ttm[it])
    #print(mV0)九个不同ttm的S0
    a[it]=(tmpcp[minKIdx, 0])#九个不同ttm的Strike price
    b[it]=mV0 #spot price





b=np.array(b)
b=b.astype(np.float32)
#newB=np.linspace(min(b),max(b),30)
#a=np.linspace(min(a),max(a),30)
a=np.array(a)
a=a.astype(np.float32)
ttm=ttm.astype(np.float32)
#ttm=np.hstack((ttm,ttm))
#newTtm = np.linspace(min(ttm), max(ttm), 30)
for i in range(9): #制定strike
    k=a[i]
    s=b[i]
    for j in range (9):#制定ttm
      sa.SetVar(s,ttm[j],cc,0.0)
      vol[i][j]=sa.SABR_func(k,0.8,0.9,0.3,0.15)

intplf = interpolate.interp2d(a, ttm, vol, kind='cubic')
newStk = np.linspace(min(a), max(a), 15);
newTtm = np.linspace(min(ttm), max(ttm), 15);
y2 = intplf(newStk, newTtm)
K, T = np.meshgrid(a, ttm);
fig = plt.figure(figsize=(12, 7));
ax = fig.gca(projection='3d');
surf = ax.plot_surface(K, T, vol, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True);
ax.set_xlabel('strike');
ax.set_ylabel('maturity');
ax.set_zlabel('Fitting vol');
plt.title('Implied vol surface fitted SABR')
plt.show()

newK, newT = np.meshgrid(newStk, newTtm);
# init
dx = np.zeros_like(y2);
dT = np.zeros_like(y2);
dxx = np.zeros_like(y2);
d = np.zeros_like(y2);

h = np.diff(newStk);  # spatial steps
k = np.diff(newTtm);  # temporal steps
# find derivatives
for i in range(1, len(newTtm) - 1):
    for j in range(1, len(newStk) - 1):
        dx[i, j] = (-h[j] / (h[j - 1] * (h[j - 1] + h[j])) * y2[i, j - 1]
                    + (h[j] - h[j - 1]) / (h[j - 1] * h[j]) * y2[i, j]
                    + h[j - 1] / (h[j] * (h[j - 1] + h[j])) * y2[i, j + 1]);
        dxx[i, j] = 2.0 * (y2[i, j - 1] / (h[j - 1] * (h[j - 1] + h[j]))
                           - y2[i, j] / (h[j - 1] * h[j])
                           + y2[i, j + 1] / (h[j] * (h[j - 1] + h[j])));
        dT[i, j] = (-k[i] / (k[i - 1] * (k[i - 1] + k[i])) * y2[i - 1, j]
                    + (k[i] - k[i - 1]) / (k[i - 1] * k[i]) * y2[i, j]
                    + k[i - 1] / (k[i] * (k[i - 1] + k[i])) * y2[i + 1, j]);
        d[i, j] = ((np.log(mV0 / newStk[j]) + (cc + 0.5 * y2[i, j] ** 2) *
                    newTtm[i]) / (y2[i, j] * np.sqrt(newTtm[i])));

# init
lovol = np.zeros_like(y2);

# By Dupire equation,
for i in range(len(newTtm)):
    for j in range(len(newStk)):
        lovol[i, j] = ((y2[i, j] ** 2 + 2.0 * y2[i, j] * newTtm[i] *
                        (dT[i, j] + (cc) * newStk[j] * dx[i, j])) / ((1.0 + newStk[j] * d[i, j] * dx[i, j] *
                                                                      np.sqrt(newTtm[i])) ** 2 + y2[i, j] * newStk[
                                                                         j] ** 2 * newTtm[i] * (dxx[i, j] - d[i, j] *
                                                                                                dx[i, j] ** 2 * np.sqrt(
                    newTtm[i]))));

# linear extrapolation at every boundary
lovol[:, 0] = 2.0 * lovol[:, 1] - lovol[:, 2];
lovol[:, -1] = 2.0 * lovol[:, -2] - lovol[:, -3];
lovol[0, :] = 2.0 * lovol[1, :] - lovol[2, :];
lovol[-1, :] = 2.0 * lovol[-2, :] - lovol[-3, :];
lovol = np.sqrt(lovol);

# plot local vol surface
fig = plt.figure(figsize=(12, 7));
ax = fig.gca(projection='3d');
surf = ax.plot_surface(newK, newT, lovol, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True);
ax.set_xlabel('strike');
ax.set_ylabel('maturity');
ax.set_zlabel('local vol');
plt.title('Local vol surface')
plt.show()
