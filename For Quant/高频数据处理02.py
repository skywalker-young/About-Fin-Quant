###处理IC数据
import os
import numpy as np
import pandas as pd
import datetime
parent_path="D:\cleaningdata"

root=[]
files=[]
for r,dirs,f in os.walk(parent_path):
    # print('root',r)
    root.append(r)
    # print('dirs',dirs)
    files.append(f)#储存IC

root=root[0:1]
files=files[0][0:2]
print(files)
# exit()
# for i in root:
start=0
end=0
step=117
timespan = np.linspace(0, 59, step)
for j in files:

    open_path=parent_path+"\\"+j

    t1=pd.read_csv(open_path,encoding="gbk")
    t1=t1[['时间','最新']]
    t1['时间']=pd.DataFrame(t1['时间'],dtype='datetime64')
    t1=t1.rename(columns={'时间':'time'})
    for a in range(0,100):
        test=t1['time'][a].minute
        if test==30:
            start=a
            break
    for b in range(len(t1) - 1, 1, -1):
        test = t1['time'][b].minute
        if test == 59:
            end = b
            break
    ##IC数据比较规范，没有中午休盘时间段数据夹杂
    raw_data=t1[start:end]
    raw_data=raw_data.reset_index(drop=True)
    # print(raw_data.tail())
    # exit()
    beginfrom = raw_data['time'][0]
    empty=pd.DataFrame()
    tttime=pd.DataFrame()
    tttime_afternoon=pd.DataFrame()
    record_morning=0
    record_afternoon_from=0
    for c in range(len(raw_data)):
        tmp=raw_data['time'][c]
        timestamp=tmp-beginfrom
        # print('tmp',tmp)
        # print('begin',beginfrom)
        # print('stamp',timestamp.seconds)

        if 1.04 > timestamp.seconds / 60 >= 1.0:
            # print('count',c)
            mean_value = np.mean(raw_data['最新'][record_morning:c])
            cal=len(raw_data['最新'][record_morning:c])
            tofill = np.full((step, 1), np.nan)

            if cal==step:
                tofill=list(raw_data['最新'][record_morning:c])
            elif cal>step:
                tmp=raw_data['最新'][record_morning:c]
                tmp=tmp[:step]
                tofill=list(tmp)
            elif cal<step and cal!=1:
                dif=step-cal
                tmp=list(raw_data['最新'][record_morning:c])

                for i in range(dif):
                    tmp.append(mean_value)
                # print('tmp',len(tmp))
                tofill=tmp
            tofill=pd.DataFrame(tofill)
            tofill=tofill.rename(columns={'最新':1})
            record_morning = c#####先把记录往下移一分钟
            beginfrom = raw_data['time'][record_morning] ##新的一分钟的时间
            # print('from',beginfrom)
            empty= pd.concat([empty,tofill],axis=0)
            # print('len empty',len(empty))
            empty=empty.reset_index(drop=True)
            tmp_fortttime = pd.DataFrame([beginfrom + datetime.timedelta(seconds=i) for i in timespan])
            tttime=pd.concat([tttime,tmp_fortttime],axis=0)
            tttime=tttime.reset_index(drop=True)

        elif 1.52>timestamp.seconds / 3600 > 1.5:
                record_afternoon_from = c
                print('aaaaaaaaaa',timestamp.seconds/3600)
                print('index',c)
                # exit()
                break
    beginfrom = raw_data['time'][record_afternoon_from]

    empty_afternoon=pd.DataFrame()
    for f in range(record_afternoon_from, len(raw_data)):
            tmp = raw_data['time'][f]
            timestamp = tmp - beginfrom
            # print('afternoon tmp', tmp)
            # print('afternoon',beginfrom)
            # print('stamp afternoon', timestamp.seconds)

            if 1.04 > timestamp.seconds / 60 >= 0.96:  # 处理下午，从13:00开始
                mean_value = np.mean(raw_data['最新'][record_afternoon_from:f])
                cal = len(raw_data['最新'][record_afternoon_from:f])
                tofill_afternoon = np.full((step, 1), np.nan)

                if cal == step:
                    tofill = list(raw_data['最新'][record_afternoon_from:f])
                elif cal > step:
                    tmp = raw_data['最新'][record_afternoon_from:f]
                    tmp = tmp[:step]
                    tofill = list(tmp)
                elif cal < step and cal != 1:
                    dif = step - cal
                    tmp = list(raw_data['最新'][record_afternoon_from:f])

                    for i in range(dif):
                        tmp.append(mean_value)
                    # print('tmp',len(tmp))
                    tofill_afternoon = tmp

                # print('fill value',mean_value)
                # exit()
                tofill_afternoon = pd.DataFrame(tofill_afternoon)
                record_afternoon_from = f
                # exit()
                beginfrom = raw_data['time'][record_afternoon_from]

                empty_afternoon = pd.concat([empty_afternoon, tofill_afternoon], axis=0)
                empty_afternoon = empty_afternoon.reset_index(drop=True)  # 下午的数据
                # print('empty_afternoon',len(empty_afternoon))
                tmp_fortttime = pd.DataFrame([beginfrom + datetime.timedelta(seconds=i) for i in timespan])
                # print('tmp',tmp_fortttime)
                tttime_afternoon = pd.concat([tttime_afternoon, tmp_fortttime], axis=0)
                tttime_afternoon = tttime_afternoon.reset_index(drop=True)
    # exit()
    empty=pd.concat([empty,empty_afternoon],axis=0)
    tttime=pd.concat([tttime,tttime_afternoon],axis=0)
    empty=empty.rename(columns={0:'price'})
    tttime=tttime.rename(columns={0:'time'})
    result=pd.concat([tttime,empty],axis=1)
    # result=result.rename(columns={0:'time',1:'price'})
    print('length of file',len(result))
    result.to_csv('D:\\cleaningdata\\tmp'+'\\'+os.path.splitext(j)[0]+'expand.csv',index=False)


