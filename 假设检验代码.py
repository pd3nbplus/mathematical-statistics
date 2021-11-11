# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:14:22 2021

@author: 潘登
"""

# %% 单样本t检验
import pandas as pd
from scipy import stats as ss
#检验期末成绩的均值是否为25
path = '统计编程/attend.xls'
data = pd.read_excel(path)
data.columns
data['final'].plot(kind = 'hist')
ss.ttest_1samp(data.final,25)
#得出p值为1e-6 <  0.05 故拒绝原假设
ss.ttest_1samp(data.final,26)
#p接近于0.547 > 0.05，故有95%的把握认为均值为26
# %% 双总体t检验
# 分析出勤人数对期末成绩有没有显著影响
data.describe()
# 将数据集划分成两部分
attend_min = data[data['attend'] <= 28]
attend_max = data[data['attend'] > 28]

#先检验方差齐性
ss.levene(attend_min.final,attend_max.final)
#得出p值小于0.05，说明方差不齐

# 双总体检验
ss.ttest_ind(attend_min.final,attend_max.final,equal_var=False)
#得出p值小于0.05，拒绝原假设， 说明两总体的均值不同
# %% 相关系数检验
from scipy.stats import pearsonr
r,p = pearsonr(data.ACT, data.termGPA)
print('相关系数为:', r, '\np-value:', p)
# %% 方差齐性检验
# 大一大二的同学的出勤方差不变
# 将数据集划分成两部分
frosh = data[data['frosh'] == 1] # 大一的
soph = data[data['soph'] == 1] # 大二的

# 检验方差齐性
ss.levene(frosh.attend,soph.attend)
# 得出p值为0.44 > 0.05，说明方差相等
# %% 单因素方差分析
# 对skipped进行离散化, 分为3个区间
data['skipped'].plot(kind = 'hist')
bins = [0, 10, 20 ,31]
labels = ['skipped_A', 'skipped_B', 'skipped_C']
data['skipped_1'] = pd.cut(data.skipped, bins, right=False, labels=labels)

skipped_A = data[data['skipped_1']=='skipped_A']
skipped_B = data[data['skipped_1']=='skipped_B']
skipped_C = data[data['skipped_1']=='skipped_C']

#检验方差齐性
ss.levene(skipped_A.final, skipped_B.final, skipped_C.final)
# 得出p值为0.11 > 0.05，说明方差相等

# 方差相等, 再做单因素方差分析
ss.f_oneway(skipped_A.final, skipped_B.final, skipped_C.final)
# 得出p值为0.01 < 0.05，拒绝原假设， 说明不同水平的skipped对期末成绩有影响
# %% 线性回归方程整体的显著性检验
import statsmodels.api as sm
def deal_with(x):
    if x == '.':
        return None
    else:
        return float(x)

data['hwrte'] = data['hwrte'].apply(deal_with)  # 转换数据类型
data['hwrte'].fillna(method='backfill', inplace=True) # 填充缺失值
y = data['final']
X=data.drop(columns = ['final', 'stndfnl', 'skipped_1'])

X=sm.add_constant(X) #添加常数项
model=sm.OLS(y,X)
results=model.fit()
y_pred=pd.DataFrame(model.predict(results.params,X),
                    columns=['pred'])
print(results.summary())
# %%  卡方检验
# 对ACT进行离散化
data['new_ACT'] = pd.cut(data.ACT,bins = 4)
# 卡方的列联表
t = pd.crosstab(data['new_ACT'],data['frosh'])
#期望频数--> 指的是如果原假设正确， 那么连列联表应该是这样， 
# 现在就是去检验到底期望与实际的差距是由误差导致的还是 ACT会收到年级的影响
ss.contingency.expected_freq(t)

ss.contingency.chi2_contingency(t,False)
# False表示不做卡方统计量的修正
# 若为真，则*和*自由度为1，应用耶茨修正，对于连续性。
# 调整的效果就是调整每一个观测值与相应的期望值相差0.5。

# 得出p值为0.005 < 0.05，拒绝原假设， 说明不同年级的学生的ACT不相等
#%% K-S检验
# 检验期末考试成绩是否服从正态分布
# 绘图查看
data['final'].plot(kind = 'hist')
ss.kstest(data['final'], 'norm')
# 得出p值为0.00< 0.05，拒绝原假设， 说明期末考试成绩不服从正态分布

# 两独立样本K-S检验
# 检验大一的与大二的同学的期末考试成绩是否同分布
ss.ks_2samp(frosh.final,soph.final)
# 得出p值为0.046< 0.05，拒绝原假设，大一的与大二的同学的期末考试成绩不同分布
# %% 游程检验
from statsmodels.sandbox.stats.runs import runstest_1samp
import random
random.seed(45)
# 生成随机变量序列
x = []
for i in range(100):
    if random.random() < 0.4:
        x.append(1)
    else:
        x.append(0)
# 做游程检验
runstest_1samp(x)
# 得出p值为0.33 > 0.05，不拒绝原假设，该序列变量值出现是随机的












