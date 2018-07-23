# -*- coding:utf-8 -*-
#使用逻辑回归（对数几率回归）对鸢尾花数据集（iris）进行分类
#4个特征变量，1个类别变量。iris每个样本都包含了4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，以及1个类别变量（label）

# Sigmoid曲线:
import matplotlib.pyplot as plt
import numpy as np
'''plt不显示中文：默认字体中没有中文字体'''
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def plot_sigmoid():
    x = np.arange(-10, 10, 0.1)
    h = Sigmoid(x)  # Sigmoid函数
    plt.plot(x, h)
    plt.axvline(0.0, color='k')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0,  0.5, 1.0])  # y axis label
    plt.title(r'Sigmoid函数曲线', fontsize = 15)
    plt.text(5,0.8,r'$y = \frac{1}{1+e^{-z}}$', fontsize = 18)
    plt.show()
#sklearn内置了IRIS数据集
# 导入所需要的包

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from plotly.offline import init_notebook_mode, iplot,plot
init_notebook_mode(connected = True)

from sklearn.datasets import load_iris #导入IRIS数据集

iris = load_iris()  #特征矩阵
pass
# 选择归一化对数据进行无量纲化处理
from sklearn.preprocessing import Normalizer
# Normalizer().fit_transform(iris.data)
target = iris.target
iData = iris.data
labels = [0,1,2]
values = [sum(target==0),sum(target==1),sum(target==2)]
# labels = data.groupby('Species').size().index
# values = data.groupby('Species').size()
trace = go.Pie(labels=labels, values=values)
layout = go.Layout(width=350, height=350)
fig = go.Figure(data=[trace], layout=layout)
# plot(fig,filename='Pie')#画饼图
#柱形图
# Feature Plot
#转换成dataframe
newA = np.concatenate([iData, target[:,np.newaxis]],axis=1)
data = pd.DataFrame(newA,columns=['sepal_length','sepal_width','petal_length','petal_width','Species'])
groups = data.groupby(by = "Species")
means, sds = groups.mean(), groups.std()
means.plot(yerr = sds, kind = 'bar', figsize = (9, 9), table = True)
# plt.show()#平均值和标准差
#散点图 四个特征两两组合对比
col_map = {0: 'orange', 1: 'green', 2: 'pink'}
pd.tools.plotting.scatter_matrix(data.loc[:, 'sepal_length':'petal_width'], diagonal = 'kde', color = [col_map[lb] for lb in data['Species']], s = 75, figsize = (11, 6))
plt.show()
# 创建训练集与测试集
# 先前取两列数据（即特征花萼长度与宽度）进行对数几率回归的分类
from sklearn.model_selection import train_test_split
X = iris.data[:, :2]             # 取前两列数据
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
# x_train.shape,y_train.shape,x_test.shape, y_test.shape
trace = go.Scatter(x = X[:,0], y = X[:,1], mode = 'markers',
                    marker = dict(color = np.random.randn(150),size = 10, colorscale='Viridis',showscale=False))
layout = go.Layout(title = '训练点', xaxis=dict(title='花萼长度 Sepal length', showgrid=False),
                    yaxis=dict(title='花萼宽度 Sepal width',showgrid=False),
                    width = 700, height = 380)
fig = go.Figure(data=[trace], layout=layout)
# plot(fig,filename='trainDot.html')#两个特征的训练集散点图
'''
导入模型，调用逻辑回归LogisticRegression()函数。
penalty: 正则化选择参数（惩罚项的种类），默认方式为L2正则化
C: 正则项系数的倒数
solver: 对于多分类任务， 使用‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ 来解决多项式loss
multi_class: 默认值‘ovr’适用于二分类问题，对于多分类问题，用‘multinomial’在全局的概率分布上最小化损失
训练LogisticRegression分类器
调用fit(x,y)的方法来训练模型，其中x为数据的属性，y为所属类型。
利用训练得到的模型对数据集进行预测 predict()，返回预测结果。
'''
from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C = 1e5) # C: Inverse of regularization strength
lr = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')
lr.fit(x_train,y_train)
print("Logistic Regression模型训练集的准确率：%.3f" %lr.score(x_train, y_train))
print("Logistic Regression模型测试集的准确率：%.3f" %lr.score(x_test, y_test))
from sklearn import metrics
y_hat = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat) #错误率，也就是np.average(y_test==y_pred)
print("Logistic Regression模型正确率：%.3f" %accuracy)
target_names = ['setosa', 'versicolor', 'virginica']
print(metrics.classification_report(y_test, y_hat, target_names = target_names))
# 可视化分类结果
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 # 第0列的范围
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5 # 第1列的范围
h = .02
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h)) # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = lr.predict(grid_test)                  # 预测分类值
# grid_hat = lr.predict(np.c_[x1.ravel(), x2.ravel()])
grid_hat = grid_hat.reshape(x1.shape)             # 使之与输入的形状相同
plt.figure(1, figsize=(6, 5))
# 预测值的显示, 输出为三个颜色区块，分布表示分类的三类区域
plt.pcolormesh(x1, x2, grid_hat,cmap=plt.cm.Paired)

# plt.scatter(X[:, 0], X[:, 1], c=Y,edgecolors='k', cmap=plt.cm.Paired)
plt.scatter(X[:50, 0], X[:50, 1], marker = '*', edgecolors='red', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker = '+', edgecolors='k', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1], marker = 'o', edgecolors='k', label='virginica')
plt.xlabel('花萼长度-Sepal length')
plt.ylabel('花萼宽度-Sepal width')
plt.legend(loc = 2)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.title("Logistic Regression 鸢尾花分类结果", fontsize = 15)
plt.xticks(())
plt.yticks(())
plt.grid()
plt.show()