import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import optimize
import numpy as np
import csv
from matplotlib import pyplot as plt
import math

#合并数据并过滤速度
def merge_data(folder,res):
    files_list = os.listdir(folder)
    df=pd.read_csv(os.path.join(folder,files_list[0]))
    df.to_csv(res,index=False)
    for i in range(1,len(files_list)):
        df=pd.read_csv(os.path.join(folder,files_list[i]))
        if folder=='data/train':
            if df['type'][1]=='拖网':
                df=df[df['速度']<40]
            elif df['type'][1]=='围网':
                df=df[df['速度']<30]
            elif df['type'][1]=='刺网':
                df=df[df['速度']<20]
        df.to_csv(res,index=False,header=False,mode='a+')

# merge_data('data/train','data/train_res.csv')
# merge_data('data/test','data/test_res.csv')


def f_1(x,A,B):
    return A*x+B

train_path='data/train_res.csv'
train=pd.read_csv(train_path)
test_path='data/test_res.csv'
test=pd.read_csv(test_path)

#拟合拖网直线
A1,B1=optimize.curve_fit(f_1,train[train['type']=='拖网']['x'],train[train['type']=='拖网']['y'])[0]
print("A1:",A1,"B1:",B1)
#拟合围网直线
A2,B2=optimize.curve_fit(f_1,train[train['type']=='围网']['x'],train[train['type']=='围网']['y'])[0]

train=csv.DictReader(open('data/train_res.csv','rt',encoding='utf-8'))
addDis1=open('data/train_AddDis.csv','wt',encoding='utf-8',errors="ignore",newline="")
train_AddDis=csv.writer(addDis1)
test=csv.DictReader(open('data/test_res.csv','rt',encoding='utf-8'))
addDis2=open('data/test_AddDis.csv','wt',encoding='utf-8',errors="ignore",newline="")
test_AddDis=csv.writer(addDis2)


#依次计算各个点到直线的距离
i=0
for row in train:
    # print(row.values())
    row['dis_tuo']=math.fabs((A1*eval(row['x'])-eval(row['y'])+B1)/math.pow(A1*A1+1,0.5));
    row['dis_wei']=math.fabs((A2*eval(row['x'])-eval(row['y'])+B2)/math.pow(A2*A2+1,0.5));
    if i==0:
        train_AddDis.writerow(('渔船ID','x','y','速度','方向','time','type','dis_tuo','dis_wei'))
    train_AddDis.writerow(row.values())
    i=i+1
addDis1.close()
i=0
for row in test:
    # print(row.values())
    row['dis_tuo']=math.fabs((A1*eval(row['x'])-eval(row['y'])+B1)/math.pow(A1*A1+1,0.5));
    row['dis_wei']=math.fabs((A2*eval(row['x'])-eval(row['y'])+B2)/math.pow(A2*A2+1,0.5));
    if i==0:
        test_AddDis.writerow(('渔船ID','x','y','速度','方向','time','dis_tuo','dis_wei'))
    test_AddDis.writerow(row.values())
    i=i+1
addDis2.close()

# 绘制某一渔船的轨迹
# df=pd.read_csv('data/train/50.csv')
# df.head()
# plt.plot(df['x'],df['y'])
# plt.show()
#
# df=pd.read_csv('data/train/2299.csv')
# df.head()
# plt.plot(df['x'],df['y'])
# plt.show()

#
# tuowang_data=train[train['type']=='拖网']
# #统计信息
# print("拖网")
# print(tuowang_data.describe())
# #
# weiwang_data=train[train['type']=='围网']
# #统计信息
# print("围网")
# print(weiwang_data.describe())
# #
# ciwang_data=train[train['type']=='刺网']
# #统计信息
# print("刺网")
# print(ciwang_data.describe())
#
# #各个渔船作业的时间范围
# tuowang_time=tuowang_data['time']
# print("tuowang working time:",tuowang_time.max(),"--",tuowang_time.min())
# weiwang_time=weiwang_data['time']
# print("weiwang working time:",weiwang_time.max(),"--",weiwang_time.min())
# ciwang_time=ciwang_data['time']
# print("ciwang working time:",ciwang_time.max(),"--",ciwang_time.min())

#绘制出轨迹
# plt.plot(tuowang_data['x'],tuowang_data['y'])
# plt.title('tuowang path')
# plt.show()
#
# plt.plot(weiwang_data['x'],weiwang_data['y'])
# plt.title('weiwang path')
# plt.show()
#
# plt.plot(ciwang_data['x'],ciwang_data['y'])
# plt.title('ciwang path')
# plt.show()
# #
# plt.plot(tuowang_data['速度'])
# plt.title('tuowang speed')
# plt.show()
#
# plt.plot(weiwang_data['速度'])
# plt.title('weiwang speed')
# plt.show()
#
# plt.plot(ciwang_data['速度'])
# plt.title('ciwang speed')
# plt.show()

