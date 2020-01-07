import os
import pandas as pd
import numpy as np
import datetime
pick=['512880.csv','510150.csv','512800.csv']
parent_path="D:\cleaningdata"
files=[]
root=[]
for r,dirs,f in os.walk(parent_path):
    # print('root',r)
    root.append(r)
    # print('dirs',dirs)
    files.append(f)#储存IC

root=root[1:]#保留两个子文件夹
###从子文件夹里读取pick的场内etf
# print(root[0][-2:])
# time=datetime.datetime(year=2019,month=3,day=int(root[0][-2:]),hour=11,minute=31,second=1)
# time2=datetime.datetime(year=2019,month=3,day=int(root[0][-2:]),hour=11,minute=25,second=1)
# print(time2<time)
# exit()
timespan=np.arange(0,59,0.5)#120个
for i in root:
    out=[]#用来储存所挑选的etf价格情况
    start=0
    end=0
    step=120
    midup=0
    middown=0
    comparedateup=datetime.datetime(year=2019,month=3,day=int(i[-2:]),hour=11,minute=31,second=1)
    comparedatedown=datetime.datetime(year=2019,month=3,day=int(i[-2:]),hour=12,minute=59,second=59)
    #数据源有11:31->12:59停盘时的记录，需要去除
    for j in pick:
        open_detail=i+"\\"+j
        #print(open_detail)
        t1=pd.read_csv(open_detail,encoding="gbk")
        t1=t1[['时间','最新价']]
        newname=os.path.splitext(j)[0]+'.OF'
        t1=t1.rename(columns={'最新价':newname,'时间':'time'})
        #print(len(t1))
        t1['time']=pd.DataFrame(t1['time'],dtype='datetime64')
        #print(t1.head())
        #print(t1.dtypes)
        #target=t1['time'][0]+datetime.timedelta(minutes=45)#所有场内etf的第一个数据都是从8:45开始记录，秒各不相同，
        #以分钟为判定，第一个分钟为30的必定是9:30开盘时刻然后，分钟变化的话即进入下一分钟。判断分钟内的数据是否有20个，
        #不足则用既有数据的算数平均数补足到20
        #print(target.minute)
        record_upanddown=[]
        for c in range(len(t1)):
            test=t1['time'][c].minute
            #print('test',test)
            if test==30:
                start=c       #找到第一个30，即9:30的索引，跳出循环，去掉之前的数据，构建新表
                break
        for cc in range(start,len(t1)):
            c1=t1['time'][cc]
            #print(c1)
            if comparedateup<c1<comparedatedown:
                record_upanddown.append(cc)
        midup=record_upanddown[0]
        middown=record_upanddown[-1]
        # print(record_upanddown)
        # exit()
        #从末尾往前面滚动
        for d in range(len(t1)-1,1,-1):
            test=t1['time'][d].minute
            if test==59:
                end=d
                break
        raw_data=pd.concat([t1[start:midup-1],t1[middown:end]],axis=0)
        raw_data=raw_data.reset_index(drop=True)
        # print(raw_data.head())
        # print(raw_data.tail())
        # exit()
        #以minute为分界，切开数据，每一个minute有20个数据，不足的用均值填充

        empty=pd.DataFrame()
        beginfrom=raw_data['time'][0]
        tttime = pd.DataFrame()#[beginfrom + datetime.timedelta(seconds=i) for i in timespan]
        # print(tttime)
        # exit()
        tttime_afternoon = pd.DataFrame()
        empty_afternoon = pd.DataFrame()

        record_morning=0
        record_afternoon_from=0
        record_afternoon=0
        for e in range(len(raw_data)):
            tmp = raw_data['time'][e]
            timestamp = tmp - beginfrom
            if 2>timestamp.seconds/60>=1.0:  #处理完上午beginfrom指向11:30
                mean_value = np.mean(raw_data[newname][record_morning:e])
                # print('fill value',mean_value)
                # exit()
                tofill = np.full((118, 1), np.nan)
                tofill=pd.DataFrame(tofill)
                tofill=tofill.fillna(mean_value)
                # print(tofill)
                # exit()
                record_morning=e
                #print(record_index)
                #exit()
                beginfrom=raw_data['time'][record_morning]
                empty=pd.concat([empty,tofill],axis=0)
                empty=empty.reset_index(drop=True)##上午的交易数据都存在empty里
                tmp_fortttime = pd.DataFrame([beginfrom + datetime.timedelta(seconds=i) for i in timespan])

                tttime=pd.concat([tttime,tmp_fortttime],axis=0)
                tttime=tttime.reset_index(drop=True)
                # print('empty',empty)
                # print('tttime',tttime)
                # exit()
            elif timestamp.seconds/3600>1.00:
                record_afternoon_from=e
                #print(timestamp.seconds/3600)
                break
        # print('record_afternoon from',record_afternoon_from)#2400
        beginfrom = raw_data['time'][record_afternoon_from]
        #tofill_afternoon=np.full((120,1),np.nan)
        #empty_afternoon=pd.DataFrame()
        # print('rrr',record_afternoon_from)
        for f in range(record_afternoon_from,len(raw_data)):
            tmp = raw_data['time'][f]
            timestamp = tmp - beginfrom
            if 2>timestamp.seconds/60>=1:  #处理下午，从13:00开始
                mean_value = np.mean(raw_data[newname][record_afternoon_from:f])
                # print('fill value',mean_value)
                # exit()
                tofill_afternoon = np.full((118, 1), np.nan)
                tofill_afternoon=pd.DataFrame(tofill_afternoon)
                tofill_afternoon=tofill_afternoon.fillna(mean_value)
                # print(tofill)
                # exit()
                record_afternoon_from=f
                #exit()
                beginfrom=raw_data['time'][record_afternoon_from]
                # print('beginfrom',beginfrom)#发现有13:33:04缺少13:33:01
                empty_afternoon=pd.concat([empty_afternoon,tofill_afternoon],axis=0)
                empty_afternoon=empty_afternoon.reset_index(drop=True)#下午的数据
                #print('empty_afternoon',empty_afternoon)
                tmp_fortttime = pd.DataFrame([beginfrom + datetime.timedelta(seconds=i) for i in timespan])
                # print('tmp',tmp_fortttime)
                tttime_afternoon=pd.concat([tttime_afternoon,tmp_fortttime],axis=0)
                tttime_afternoon=tttime_afternoon.reset_index(drop=True)
                # print(tttime_afternoon)

        empty=pd.concat([empty,empty_afternoon],axis=0)
        empty=empty.reset_index(drop=True)
        print(i)
        print(j)
        print(len(empty))
        tttime=pd.concat([tttime,tttime_afternoon],axis=0)
        tttime=tttime.reset_index(drop=True)
        print(len(tttime))

        empty=empty.rename(columns={0:newname})
        tttime=tttime.rename(columns={0:'time'})

        out=pd.concat([tttime,empty],axis=1)
        out.to_csv(i+'\\'+os.path.splitext(j)[0]+'handled.csv',encoding='gbk',index=False)


#发现512880在raw_data在原始数据下多了11:30:02的数据，手动删除。。。
print('finish')
