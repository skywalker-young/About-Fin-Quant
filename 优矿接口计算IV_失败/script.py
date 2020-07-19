#iVIX based on http://www.sse.com.cn/assortment/options/neovolatility/c/4206989.pdf
from CAL.PyCAL import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')
from matplotlib import dates
from pandas import Series, DataFrame, concat
from scipy import interpolate
import math
import time
from datetime import datetime

#shibor数据下载地址：http://www.shibor.org/shibor/web/DataService.jsp
#把下载来的数据表中的变量名‘日期’改成‘date’；'O/N'改成'1D'
shibor=pd.read_excel('shibor1516.xls')
shibor['date']=pd.to_datetime(shibor['date'])
shibor=shibor.set_index('date')
shibor_rate=shibor
shibor_rate.head()


def histDayDataOpt50ETF(date):
    date_str = date.toISO().replace('-', '')

    #使用DataAPI.OptGet，拿到已退市和上市的所有期权的基本信息
    info_fields = [u'optID', u'varSecID', u'varShortName', u'varTicker', u'varExchangeCD', u'varType', 
                   u'contractType', u'strikePrice', u'contMultNum', u'contractStatus', u'listDate', 
                   u'expYear', u'expMonth', u'expDate', u'lastTradeDate', u'exerDate', u'deliDate', 
                   u'delistDate']
    opt_info = DataAPI.OptGet(optID='', contractStatus=[u"DE",u"L"], field=info_fields, pandas="1")
    opt_info = opt_info[opt_info.varSecID == "510050.XSHG"]

    #使用DataAPI.MktOptdGet，拿到历史上某一天的期权成交信息
    mkt_fields = [u'ticker', u'optID', u'secShortName', u'exchangeCD', u'tradeDate', u'preSettlePrice', 
                  u'preClosePrice', u'openPrice', u'highestPrice', u'lowestPrice', u'closePrice', 
                  u'settlPrice', u'turnoverVol', u'turnoverValue', u'openInt']
    opt_mkt = DataAPI.MktOptdGet(tradeDate=date_str, field=mkt_fields, pandas = "1")
    opt_mkt = opt_mkt[opt_mkt.exchangeCD == "XSHG"]

    opt_info = opt_info.set_index(u"optID")
    opt_mkt = opt_mkt.set_index(u"optID")
    opt = pd.concat([opt_info, opt_mkt], axis=1, join='inner').sort_index()
    return opt
    
    
def periodsSplineRiskFreeInterestRate(options, date):
    
    exp_dates = map(Date.parseISO, np.sort(options.expDate.unique()))
    periods = {}
    for epd in exp_dates:
        periods[epd] = (epd - date)*1.0/365.0
    
    date_str = date.toISO()
    if pd.to_datetime(date_str) >= shibor_rate.index[-1]:
        date_str = shibor_rate.index[-1]
        shibor_values = shibor_rate.ix[-1].values
    else:
        shibor_values = shibor_rate[date_str].values[0]
        
    shibor = {}
    period = np.asarray([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 360.0]) / 360.0
    min_period = min(period)
    max_period = max(period)
    for p in periods.keys():
        tmp = periods[p]
        if periods[p] > max_period:
            tmp = max_period * 0.99999
        elif periods[p] < min_period:
            tmp = min_period * 1.00001
        # 此处使用SHIBOR来插值
        sh = interpolate.spline(period, shibor_values, tmp, order=3)
        shibor[p] = sh/100.0
    return shibor
    
    
def calDayVIX(date, opt_info):    
    var_sec = u"510050.XSHG"
    # 使用DataAPI.MktOptdGet，拿到历史上某一天的期权行情信息
    date_str = date.toISO().replace('-', '')
    fields_mkt = [u"optID", "tradeDate", "closePrice", 'settlPrice']
    opt_mkt = DataAPI.MktOptdGet(tradeDate=date_str, field=fields_mkt, pandas="1")
    opt_mkt = opt_mkt.set_index(u"optID")
    opt_mkt[u"price"] = opt_mkt['closePrice']

    # concat某一日行情和期权基本信息，得到所需数据
    opt = pd.concat([opt_info, opt_mkt], axis=1, join='inner').sort_index()
    opt = opt[opt.varSecID==var_sec]
    exp_dates = map(Date.parseISO, np.sort(opt.expDate.unique()))
    trade_date = date
    exp_periods = {}
    for epd in exp_dates:
        exp_periods[epd] = (epd - date)*1.0/365.0
    options= histDayDataOpt50ETF(date)
    risk_free = periodsSplineRiskFreeInterestRate(options,date)

    sigma_square = {}
    for date in exp_dates:
        # 计算某一日的vix
        opt_date = opt[opt.expDate==date.toISO()]
        rf = risk_free[date]
        #rf = 0.05
        
        opt_call = opt_date[opt_date.contractType == 'CO'].set_index('strikePrice')
        opt_put = opt_date[opt_date.contractType == 'PO'].set_index('strikePrice')
        opt_call_price = opt_call[[u'price']].sort_index()
        opt_put_price = opt_put[[u'price']].sort_index()
        opt_call_price.columns = [u'callPrice']
        opt_put_price.columns = [u'putPrice']
        opt_call_put_price = pd.concat([opt_call_price, opt_put_price], axis=1, join='inner').sort_index()
        opt_call_put_price['diffCallPut'] = opt_call_put_price.callPrice - opt_call_put_price.putPrice

        strike = abs(opt_call_put_price['diffCallPut']).idxmin()
        price_diff = opt_call_put_price['diffCallPut'][strike]
        ttm = exp_periods[date]
        fw = strike + np.exp(ttm*rf) * price_diff
        strikes = np.sort(opt_call_put_price.index.values)

        delta_K_tmp = np.concatenate((strikes, strikes[-1:], strikes[-1:])) 
        delta_K_tmp = delta_K_tmp - np.concatenate((strikes[0:1], strikes[0:1], strikes))
        delta_K = np.concatenate((delta_K_tmp[1:2], delta_K_tmp[2:-2]/2, delta_K_tmp[-2:-1]))
        delta_K = pd.DataFrame(delta_K, index=strikes, columns=['deltaStrike'])
        
        # opt_otm = opt_out_of_money
        opt_otm = pd.concat([opt_call[opt_call.index>fw], opt_put[opt_put.index<fw]], axis=0, join='inner')
        opt_otm = pd.concat([opt_otm, delta_K], axis=1, join='inner').sort_index()
        
        # 计算VIX时，比forward price低的第一个行权价被设置为参考行权价，参考值以上
        # 的call和以下的put均为虚值期权，所有的虚值期权被用来计算VIX，然而计算中发
        # 现，有时候没有比forward price更低的行权价，例如2015-07-08，故有以下关于
        # 参考行权价的设置
        strike_ref = fw
        if len((strikes[strikes < fw])) > 0:
            strike_ref = max([k for k in strikes[strikes < fw]])
            opt_otm['price'][strike_ref] = (opt_call['price'][strike_ref] + opt_call['price'][strike_ref])/2.0

        exp_rt = np.exp(rf*ttm)
        opt_otm['sigmaTerm'] = opt_otm.deltaStrike*opt_otm.price/(opt_otm.index)/(opt_otm.index)
        sigma = opt_otm.sigmaTerm.sum()
        sigma = (sigma*2.0*exp_rt - (fw*1.0/strike_ref - 1.0)**2)/ttm
        sigma_square[date] = sigma

    # d_one, d_two 将被用来计算VIX(30):
    if (exp_periods[exp_dates[0]] >= 7.0/365.0) & (exp_periods[exp_dates[0]] < 30.0/365.0):
        d_one = exp_dates[0]
        d_two = exp_dates[1]
        w = (exp_periods[d_two] - 30.0/365.0)/(exp_periods[d_two] - exp_periods[d_one])
        vix30 = exp_periods[d_one]*w*sigma_square[d_one] + exp_periods[d_two]*(1 - w)*sigma_square[d_two]
        vix30 = 100*np.sqrt(vix30*365.0/30.0)
    if exp_periods[exp_dates[0]] >= 30.0/365.0:
        d_one = exp_dates[0]
        vix30 = sigma_square[d_one]
        vix30 = 100*np.sqrt(vix30)
    if (exp_periods[exp_dates[0]] < 7.0/365.0) & (exp_periods[exp_dates[1]] < 30.0/365.0):
        d_one = exp_dates[1]
        d_two = exp_dates[2]
        w = (exp_periods[d_two] -30.0/365.0)/(exp_periods[d_two] - exp_periods[d_one])
        vix30 = exp_periods[d_one]*w*sigma_square[d_one] + exp_periods[d_two]*(1 - w)*sigma_square[d_two]
        vix30 = 100*np.sqrt(vix30*365.0/30.0)
    if (exp_periods[exp_dates[0]] < 7.0/365.0) & (exp_periods[exp_dates[1]] >= 30.0/365.0):
        d_one = exp_dates[1]
        vix30 = sigma_square[d_one]
        vix30 = 100*np.sqrt(vix30)
    # d_one, d_two 将被用来计算VIX(60):
    d_one = exp_dates[1]
    d_two = exp_dates[2]
    w = (exp_periods[d_two] - 60.0/365.0)/(exp_periods[d_two] - exp_periods[d_one])
    vix60 = exp_periods[d_one]*w*sigma_square[d_one] + exp_periods[d_two]*(1 - w)*sigma_square[d_two]
    vix60 = 100*np.sqrt(vix60*365.0/60.0)

    return vix30, vix60
    
def getHistDailyVIX(beginDate, endDate):
    # 计算历史一段时间内的VIX指数并返回
    optionVarSecID = u"510050.XSHG"
    
    # 使用DataAPI.OptGet，一次拿取所有存在过的期权信息，以备后用
    fields_info = ["optID", u"varSecID", u'contractType', u'strikePrice', u'expDate']
    opt_info = DataAPI.OptGet(optID='', contractStatus=[u"DE", u"L"], field=fields_info, pandas="1")
    opt_info = opt_info.set_index(u"optID")
    
    cal = Calendar('China.SSE')
    cal.addHoliday(Date(2015,9,3))
    cal.addHoliday(Date(2015,9,4))
    
    dates = cal.bizDatesList(beginDate, endDate)
    histVIX = pd.DataFrame(0.0, index=map(Date.toDateTime, dates), columns=['VIX30','VIX60'])
    histVIX.index.name = 'tradeDate'
    for date in histVIX.index:
        try:
            vix30, vix60 =  calDayVIX(Date.fromDateTime(date), opt_info)
        except:
            histVIX = histVIX.drop(date)
            continue
        histVIX['VIX30'][date] = vix30
        histVIX['VIX60'][date] = vix60
    return histVIX

def getHistOneDayVIX(date):
    # 计算历史某天的VIX指数并返回
    optionVarSecID = u"510050.XSHG"
    
    # 使用DataAPI.OptGet，一次拿取所有存在过的期权信息，以备后用
    fields_info = ["optID", u"varSecID", u'contractType', u'strikePrice', u'expDate']
    opt_info = DataAPI.OptGet(optID='', contractStatus=[u"DE", u"L"], field=fields_info, pandas="1")
    opt_info = opt_info.set_index(u"optID")
    
    cal = Calendar('China.SSE')
    cal.addHoliday(Date(2015,9,3))
    cal.addHoliday(Date(2015,9,4))
    
    if cal.isBizDay(date):
        vix30, vix60 = 0.0, 0.0
        vix30, vix60 =  calDayVIX(date, opt_info)
        return vix30, vix60
    else:
        print date, "不是工作日"
        
#示例：计算某一交易日的iVIX
date=Date(2015,3,19)
getHistOneDayVIX(date)

#示例：计算某交易日区间的iVIX
begin=Date(2018,1,1)
end=Date(2018,12,31)
hist_VIX=getHistDailyVIX(beginDate=begin, endDate=end)
    
