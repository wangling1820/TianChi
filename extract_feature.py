import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from tqdm import tqdm
import lightgbm as lgb
from matplotlib import pyplot as plt

def group_feature(df,key,target,aggs):
    agg_dict={}
    for ag in aggs:
        agg_dict[f'{target}_{ag}']=ag
    # print(agg_dict)
    t=df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def extract_feature(df,train):
    t=group_feature(df,'ship','x',['max','min','mean','std','skew','sum'])
    train=pd.merge(train,t,on='ship',how='left')
    t=group_feature(df,'ship','x',['count'])
    train=pd.merge(train,t,on='ship',how='left')
    t=group_feature(df,'ship','y',['max','min','mean','std','skew','sum'])
    train=pd.merge(train,t,on='ship',how='left')
    t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t=group_feature(df,'ship','dis_tuo',['max','min','mean','std','skew','sum'])
    train=pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship', 'dis_wei', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    train['x_max-x_min']=train['x_max']-train['x_min']
    train['y_max-y_min']=train['y_max']-train['y_min']
    train['y_max-x_min']=train['y_max']-train['x_min']
    train['x_max-y_min']=train['x_max']-train['y_min']
    #斜率
    train['slope']=train['y_max-y_min']/np.where(train['x_max-x_min']==0,0.001,train['x_max-x_min'])
    train['area']=train['x_max-x_min']*train['y_max-y_min']

    return train


train=pd.read_csv('data/train_AddDis.csv')
train.columns =['ship','x','y','v','d','time','type','dis_tuo','dis_wei']
test=pd.read_csv('data/test_AddDis.csv')
test.columns=['ship','x','y','v','d','time','dis_tuo','dis_wei']

train_label=train.drop_duplicates('ship')
test_label=test.drop_duplicates('ship')

#查看各个类型占得比重
# print(train_label['type'].value_counts(1))

type_map=dict(zip(train_label['type'].unique(),np.arange(3)))
type_map_rev={v:k for k,v in type_map.items()}
print(type_map_rev)
train_label['type']=train_label['type'].map(type_map)
# test_label['type']=test_label['type'].map(type_map)

train_label=extract_feature(train,train_label)
test_label=extract_feature(test,test_label)

features=[x for x in train_label.columns if x not in ['ship','type','time']]
target='type'
# print(len(features),',',features)

params={
    'n_estimators':5000,
    'boosting_type':'gbdt',
    'objective':'multiclass',
    'num_class':3,
    'early_stopping_rounds':100,
    'learning_rate':0.05,
}

fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
X=train_label[features].copy()
y=train_label[target]
models=[]
pred=np.zeros((len(test_label),3))
oof=np.zeros((len(X),3))
for index,(train_idx,val_idx) in enumerate(fold.split(X,y)):
    train_set=lgb.Dataset(X.iloc[train_idx],y.iloc[train_idx])
    val_set=lgb.Dataset(X.iloc[val_idx],y.iloc[val_idx])

    model=lgb.train(params,train_set,valid_sets=[train_set,val_set],verbose_eval=100)
    models.append(model)
    val_pred=model.predict(X.iloc[val_idx])
    oof[val_idx]=val_pred
    val_y=y.iloc[val_idx]
    val_pred=np.argmax(val_pred,axis=1)
    print(index,'val f1:',metrics.f1_score(val_y,val_pred,average='macro'))

    test_pred=model.predict(test_label[features])
    pred+=test_pred/10

oof = np.argmax(oof,axis=1)
print('oof f1',metrics.f1_score(oof,y,average='macro'))

pred=np.argmax(pred,axis=1)
sub=test_label[['ship']]
sub['pred']=pred

print(sub['pred'].value_counts(1))
sub['pred']=sub['pred'].map(type_map_rev)
sub.to_csv('data/result.csv',index=None,header=None)

ret=[]
for index,model in enumerate(models):
    df=pd.DataFrame()
    df['name']=model.feature_name()
    df['score']=model.feature_importance()
    df['fold']=index
    ret.append(df)

df=pd.concat(ret)
df=df.groupby('name',as_index=False)['score'].mean()
df=df.sort_values(['score'],ascending=False)
print(df)






