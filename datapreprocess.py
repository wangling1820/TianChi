"""
 * @Author: wangling 
 * @Date: 2020-01-10 18:53:09 
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
def merge_data(folder, res):
    files_list = os.listdir(folder)
    df = pd.read_csv(os.path.join(folder, files_list[0]))
    df.to_csv(res, index=False)
    for i in range(1, len(files_list)):
        df = pd.read_csv(os.path.join(folder, files_list[i]))
        df.to_csv(res, index=False, header=False, mode='a+')


merge_data('./data/train', './data/trian_res.csv')
merge_data('./data/test', './data/test_res.csv')



df = pd.read_csv('./data/train/50.csv')
df.head()

plt.plot(df['x'], df['y'])
plt.show()


df = pd.read_csv('./data/train/2299.csv')
df.head()

plt.plot(df['x'], df['y'])
plt.show()

train_data = './data/trian_res.csv'
train = pd.read_csv(train_data)
test_data = './data/test_res.csv'
test = pd.read_csv(test_data)


tuowang_data = train[train['type'] == '拖网']
#  统计信息
print('拖网')
print(tuowang_data.describe())


weiwang_data = train[train['type'] == '围网']
# 统计信息
print('围网')
print(weiwang_data.describe())

ciwang_data = train[train['type'] == '刺网']
#  统计信息
print('刺网')
print(ciwang_data.describe())

plt.plot(tuowang_data['x'], tuowang_data['y'])
plt.title('tuo wang lu xian tu')
plt.show()

plt.plot(tuowang_data['速度'])
plt.title('tuo wang de su du')
plt.show()

plt.plot(weiwang_data['速度'])
plt.title('wei wang de su du')
plt.show()

plt.plot(ciwang_data['速度'])
plt.title('ci wang de su du')
plt.show()


