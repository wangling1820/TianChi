import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


train = pd.read_csv('./data/train_feature.csv')
train.head()

label2num = {'拖网':0, '刺网':1, '围网':2}
train['type'] = train['type'].map(label2num)
label = np.array(train['type'])

x = np.array(train.drop('type', axis=1))

param = {'num_leaves': 31,
#          'min_data_in_leaf': 30, 
         'objective':'multiclassova',
         'num_class':3,
         'learning_rate': 0.01,
#          "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
#          "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
#          "bagging_seed": 11,
#          "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1
        }
folds = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(folds.split(x, label)):
    print('fold = ', fold+1)
    curr_train_data = lgb.Dataset(x[train_idx], label[train_idx])
    curr_val_data = lgb.Dataset(x[val_idx], label[val_idx])
    num_boost_round = 1000
    booster_model = lgb.train(param, 
                                curr_train_data, 
                                num_boost_round, 
                                valid_sets = [curr_val_data], 
                                verbose_eval=500,
                                early_stopping_rounds = 1000)
    label_pred = booster_model.predict(x[val_idx], num_iteration=booster_model.best_iteration)
    label_pred_encode = np.argmax(label_pred, axis=1)
    print(fold + 1, '_validation f1_score is ', f1_score(label_pred_encode, label[val_idx], average='macro'))

test_fea = pd.read_csv('./data/test_feature.csv')
test_fea
test = np.array(test_fea.drop('id', axis=1))

test_pred = booster_model.predict(test, num_iteration=booster_model.best_iteration)
test_pred_encode = np.argmax(test_pred, axis=1)
len(test_pred_encode)

res = pd.DataFrame( {'id':test_fea['id'], 'type':test_pred_encode})  # 生成padans数据格式
print(res) 
num2label = {0: '拖网', 1 :'刺网', 2: '围网'}
res['type'] = res['type'].map(num2label)
res.to_csv('./data/res_001.csv', header=False, index=False)
print('done!')