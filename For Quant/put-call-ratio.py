import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
pd.set_option('display.width',None)
pro=ts.pro_api('14a5924a30fb5cc5aac640e94eb72a9c9fdc8da4652991f9e98f8914')


TradingDay=pro.trade_cal(exchange='SSE',start_date='20190523',end_date='20190603',is_open=1)#0415,0416ok
TradingDay=TradingDay['cal_date']

basicfields=['ts_code','call_put','exercise_price','maturity_date','name']#name没必要
OptInfo=pro.opt_basic(exchange='SSE',fields=basicfields)
#print(OptInfo)
data=pro.opt_daily( trade_date=TradingDay[0],exchange='SSE')
for day in TradingDay[1:]:
    tmp=pro.opt_daily(trade_date=day ,exchange='SSE')
    data=pd.merge(data,tmp,how='outer')
#print(data)

options=pd.merge(data,OptInfo,how='left')
options=options.sort_values(by=['trade_date','exercise_price'])
#print(options)
#oi 持仓量
hold=options.loc[(options['exercise_price']==2.800) & (options['maturity_date']=='20190626')]

hold=hold.reset_index(drop=True)
print(hold)
length=len(hold)
tmp=[]
for i in range(0,length,2):
    tmp.append(hold.ix[i]['oi'])

plt.plot(tmp)
plt.show()
