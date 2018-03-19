def SetVar(input_s, input_Tau, input_r, input_q):
    global s, Tau, r, q;
    s = input_s;
    Tau = input_Tau;
    r = input_r;
    q = input_q;


# def GetVar():
#    global g_s, g_Tau, g_r, g_q;
#    return g_s, g_Tau, g_r, g_q;

def SABR_func(K, alp, bet, rho, nu):
    import numpy as np;
    #    from math import isnan, exp, log, sqrt;
    #    s, Tau, r, q

    f = s * np.exp((r - q) * Tau);

    z = nu / alp * (f * K) ** (0.5 * (1 - bet)) * np.log(f / K);
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho));

    zdivxz = z / xz;

    # exception cases
    zdivxz[np.isnan(zdivxz)] = 1.0;

    result = (alp * (f * K) ** (0.5 * (bet - 1)) *
              (1 + (((1 - bet) * np.log(f / K)) ** 2 / 24 + ((1 - bet) * np.log(f / K)) ** 4 / 1920)) ** (-1.0)
              * zdivxz
              * (1 + (((1 - bet) * alp) ** 2 / (24 * (f * K) ** (1 - bet))
                      + 0.25 * alp * bet * rho * nu / ((f * K) ** (0.5 * (1 - bet)))
                      + ((2 - 3 * rho ** 2) * nu ** 2) / 24) * Tau)
              );

    return result;

#######################################分割线
#####################################
$#####################################
import pandas as pd;
import numpy as np;
import bsm_functions as bsf
from datetime import datetime
from math import exp
import matplotlib.pyplot as plt
from pylab import cm
from scipy.optimize import curve_fit
from scipy import interpolate
import SABR as sa
from mpl_toolkits.mplot3d import Axes3D

def time_tango(dates):
    return datetime.strptime("{}".format(dates), "%Y-%m-%d")


convDates = 365.0;  # date convention
V0 = 2952.48;  # EUROSTOXX50 index at '2016-01-15'
qd = '2016-01-15';  # quote date
cc = 0.015;  # example cost of carry

# excel file read
xl = pd.ExcelFile("20160101_EUROSTOXX50.xlsx");

# select sheet
euro = xl.parse("Euro");

# make a float type
euro['STRIKE'] *= 1.0;

# set column name
euroTypes = euro.apply(lambda x: pd.lib.infer_dtype(x.values));
for col in euroTypes[euroTypes == 'unicode'].index:
    euro[col] = euro[col].astype(str)
euro['IMVOL'] = 0.0;
euro['TTM'] = 0.0;

# calculate time to maturity
for row in range(len(euro)):
    euro.set_value(row, 'TTM', 1.0 * (time_tango(euro.loc[row]['EXP_DATE']) - time_tango(qd)).days / convDates);

# find a unique value (remove duplicate)
expirySet = pd.Series(euro['EXP_DATE'].values.ravel()).unique();
stkSet = pd.Series(euro['STRIKE'].values.ravel()).unique();

# init
y1 = np.zeros([len(expirySet), len(stkSet)]);
ttm = np.zeros(len(expirySet));

# time slices iteration
for it in range(len(expirySet)):

    expiry = expirySet[it];  # iterate time(maturity) slice

    # select call
    sc = euro.loc[(euro['QUOTE_DATE'] == qd) & (euro['OPTION_TYPE'] == 'Call')];
    # select put
    sp = euro.loc[(euro['QUOTE_DATE'] == qd) & (euro['OPTION_TYPE'] == 'Put')];

    # time to maturity
    ttm[it] = 1.0 * (time_tango(expiry) - time_tango(qd)).days / convDates;

    # select time slice
    sc1 = sc.loc[sc['EXP_DATE'] == expiry];  # call
    sp1 = sp.loc[sp['EXP_DATE'] == expiry];  # put

    stkcp = list(set(sc1['STRIKE']) & set(sp1['STRIKE']));

    # init, tmpcp: stk | CALL price | PUT price
    tmpcp = np.zeros((len(stkcp), 3));

    for i in range(len(stkcp)):
        tmpcp[i, 0] = stkcp[i];
        tmpcp[i, 1] = sc1[sc1['STRIKE'] == stkcp[i]]['PRICE'];
        tmpcp[i, 2] = sp1[sp1['STRIKE'] == stkcp[i]]['PRICE'];

    dif = abs(tmpcp[:, 1] - tmpcp[:, 2]);  # difference

    # minimum ((near the) At The Money)
    mindif = np.min(dif);
    minK = tmpcp[dif == mindif, 0];
    minKIdx = np.where(tmpcp[:, 0] == minK)[0][0];

    # synthetic forward(S = C-P + Ke^(-cc*T)) by using Put-Call parity
    mV0 = tmpcp[minKIdx, 1] - tmpcp[minKIdx, 2] + tmpcp[minKIdx, 0] * exp(-(cc) * ttm[it]);

    # S0 = dataC(minIdx,2)-dataP(minIdx,2)...
    #    +dataC(minIdx,1)*exp(-(r-q)*Tau); % No-arbitrage condition(P-C parity)

    # find a implied vol (using Newton method)
    for i in sc1.index:
        imp_vol = bsf.bsm_call_imp_vol(mV0, sc1.loc[i]['STRIKE'], sc1.loc[i]['TTM'], cc, sc1.loc[i]['PRICE'], 1.0);
        sc1.set_value(i, 'IMVOL', imp_vol);
    for i in sp1.index:
        imp_vol = bsf.bsm_put_imp_vol(mV0, sp1.loc[i]['STRIKE'], sp1.loc[i]['TTM'], cc, sp1.loc[i]['PRICE'], 1.0);
        sp1.set_value(i, 'IMVOL', imp_vol);

    ## plot implied vol 1 time slice
    # plt.figure(1)
    # plt.figure(figsize = (10, 7));
    # plt.plot(sc1['STRIKE'], sc1['IMVOL'], label = 'Call', lw = 1.5)
    # plt.plot(sp1['STRIKE'], sp1['IMVOL'], label = 'Put', lw = 1.5)
    # plt.xlabel('STRIKE')
    # plt.ylabel('Imvol')
    # plt.legend()
    # plt.show()

    # By using synthetic forward, combine implied vol of OTM put and call
    lenc = sum(sc1['STRIKE'] <= float(minK));
    lenp = sum(sp1['STRIKE'] > float(minK));

    # init
    cp = np.zeros((2, lenc + lenp));

    # write the obtained solution
    cp[0, :lenc] = sc1[sc1['STRIKE'] <= float(minK)]['STRIKE']
    cp[0, lenc:] = sp1[sp1['STRIKE'] > float(minK)]['STRIKE']
    cp[1, :lenc] = sc1[sc1['STRIKE'] <= float(minK)]['IMVOL']
    cp[1, lenc:] = sp1[sp1['STRIKE'] > float(minK)]['IMVOL']

    # plt.figure(2)
    # plt.figure(figsize = (10, 7));
    # plt.plot(cp[0, :], cp[1, :], label = 'Call + Put', lw = 1.5)
    # plt.xlabel('STRIKE')
    # plt.ylabel('Imvol')
    # plt.legend()
    # plt.show()

    # trick for using some variables in function like global variables
    sa.SetVar(mV0, ttm[it], cc, 0.0);
    # y = sa.SABR_func(cp[0, :], 0.8, 0.9, 0.3, 0.15);

    # popt, pcov = curve_fit(func, x, yn)
    init_guess = np.array([0.5, 0.5, 0.5, 0.5]);  # initial value
    sa.SetVar(mV0, ttm[it], cc, 0.0);

    # Least square curve fit by Scipy
    # By using this function, fit implied vol curve to SABR function at one of time slices
    popt, pcov = curve_fit(sa.SABR_func, cp[0, :], cp[1, :], p0=init_guess);

    # write
    y1[it, :] = sa.SABR_func(stkSet, *popt);

    print
    ("iteration %d done..." % it)
#    plt.figure(3)
#    plt.figure(figsize = (10, 7));
#    plt.plot(cp[0, :], cp[1, :], label = 'IMVOL', lw = 1.5)
#    plt.plot(stkSet, y1, label = 'SABR', lw = 1.5)
#    plt.xlabel('STRIKE')
#    plt.ylabel('Imvol')
#    plt.legend()
#    plt.show()

# Plot implied vol surface fitted SABR
K, T = np.meshgrid(stkSet, ttm);
fig = plt.figure(figsize=(12, 7));
ax = fig.gca(projection='3d');
surf = ax.plot_surface(K, T, y1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True);
ax.set_xlabel('strike');
ax.set_ylabel('maturity');
ax.set_zlabel('Fitting vol');
plt.title('Implied vol surface fitted SABR')

plt.show()







