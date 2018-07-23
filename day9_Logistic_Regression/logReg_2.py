# -*- coding:utf-8 -*-
#何将原始文件中的数据转变成机器学习算法可用的numpy数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
iris_path = 'iris.csv'
iris = pd.read_csv(iris_path)
# print(iris.head())
def iris_type(s):
    class_label = {'setosa':0, 'versicolor':1, 'virginica':2}
    return class_label[s]
# Step 2: 将第4列内容映射至iris_type函数定义的内容,查看效果
new_iris = pd.io.parsers.read_csv(iris_path, converters = {4:iris_type})
# print(new_iris.head())
# Step 3: 将new_iris解析至numpy array
data = np.array(new_iris)  # 或者直接new_iris.values,结果是一样的
print(data[:10,:] )       # 查看前10行的数据
# Step 4:将原始数据集划分成训练集与测试集

# 用np.split按列（axis=1）进行分割
# (4,):分割位置，前4列作为x的数据，第4列之后都是y的数据
x,y = np.split(data, (4,), axis = 1)
# X = x[:,0:2] # 取前两列特征
# 用train_test_split将数据按照7：3的比例分割训练集与测试集，
# 随机种子设为1（每次得到一样的随机数），设为0或不设（每次随机数都不同）
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
'''
Pipeline(steps) 利用sklearn提供的管道机制Pipeline来实现对全部步骤的流式化封装与管理。
第一个环节：可以先进行 数据标准化 StandardScaler()
中间环节：可以加上 PCA降维处理 取2个重要特征
最终环节：逻辑回归分类器
'''
pipe_LR = Pipeline([
                    ('sc', StandardScaler()),#去均值和方差归一化。
                    ('pca', PCA(n_components = 2)),
                    ('clf_lr', LogisticRegression(random_state=1))
                    ])
# 开始训练
pipe_LR.fit(x_train, y_train.ravel())
# 分类器评估
print("训练集准确率: %0.2f" %pipe_LR.score(x_train, y_train))
print("测试集准确率: %0.2f" %pipe_LR.score(x_test, y_test))
y_hat = pipe_LR.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat)
print("逻辑回归分类器的准确率：%0.2f" % accuracy)
target_names = ['setosa', 'versicolor', 'virginica']
print(metrics.classification_report(y_test, y_hat, target_names = target_names))
'''
交叉验证常用于防止模型过于复杂而造成过拟合，同时也称为循环估计。
基本思想是将原始数据分成K组（一般是平均分组），每个子集数据分别做一次验证集或测试集，其余的K-1个子集作为训练集。
这样就会得到K个模型，取这K个模型的分类准确率的平均数作为分类器的性能指标更具说服力。
这里我们使用的是5折交叉验证(5-fold cross validation)
'''
iris_data = x
iris_target = y
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe_LR, iris_data, iris_target.ravel(), cv = 5,scoring='f1_macro') # ravel() 将y shape转变成（n_samples,）
print("5折交叉验证:\n逻辑回归分类器的准确率：%.2f 误差范围：(+/- %.2f)"%(scores.mean(), scores.std()*2))
'''
网格搜索：网格搜索就是尝试各种可能的参数组合值
试集除了用来调参，也被用来评估模型的好坏。这样做的结果会导致模型的最终评分结果会比实际要好。
对应的解决方法就是对训练集再进行一次分割：分成训练集与验证集.
'''
# 简单网格搜素 Simple Grid Search
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris_data, iris_target, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("训练集大小:{} 验证集大小:{} 测试集大小:{}".format(
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0]))
best_score = 0.0
for penalty in ['l1','l2']:
    for C in [0.01,0.1, 1, 10, 100]:
        lr_clf = LogisticRegression(C = C, penalty = penalty)
        lr_clf.fit(X_train, y_train.ravel())          # 训练
        score = lr_clf.score(X_val, y_val.ravel())    # 调参
        if score > best_score:                # 找到最好score下的参数
            best_score = score
            best_parameters = {'penalty':penalty,'C':C}
lr = LogisticRegression(**best_parameters)    #使用最佳参数，构建新的模型
lr.fit(X_trainval,y_trainval.ravel())         #使用训练集和验证集进行训练,因为数据更多效果更好
test_score = lr.score(X_test,y_test.ravel())  # evaluation模型评估
print("验证集 best score: %.2f"%(best_score))
print("最好的参数:{}".format(best_parameters))
print("测试集 best score: %.2f" %(test_score))
# 网格搜索验证 Grid Search Cross-Validation
# 结合网格搜索与交叉验证的方式
print("Start GridSearch...")
from sklearn.model_selection import GridSearchCV

param_range = [0.01,0.1, 1, 10, 100]     # 参数集合

param_grid_lr= {'C': param_range,        # 正则化系数λ的倒数,数值越小，正则化越强
                'penalty': ['l1','l2']}  # 对参数的惩罚项(约束),增强泛化能力，防止overfit
# 创建 grid search实例
clf = GridSearchCV(estimator = LogisticRegression(random_state=0), # 模型
                    param_grid = param_grid_lr,
                    scoring = 'accuracy',
                    cv = 10)                         # 10折交叉验证
# fit grid search
best_model = clf.fit(X_trainval,y_trainval.ravel())
# 查看效果最好的超参数
print("最好模型的超参数：")
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('逻辑回归模型best score:%.2f' % best_model.best_score_)
print("测试集准确率: %0.2f" %best_model.score(X_test, y_test))
'''最好模型的超参数：
Best Penalty: l1
Best C: 10
逻辑回归模型best score:0.98
测试集准确率: 0.97
'''
