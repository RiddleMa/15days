import pandas as pd
import  numpy as np

pd.DataFrame(np.random.rand(10,5)) # 创建一个5列10行的由随机浮点数组成的数据框 DataFrame
my_list = ['Kesci',100,'欢迎来到科赛网']
pd.Series(my_list) # 从一个可迭代的对象 my_list 中创建一个数据组
df = pd.DataFrame(np.random.rand(10,5))
df.index = pd.date_range('2017/1/1', periods=df.shape[0]) # 添加一个日期索引 index
df.head(3)  # 查看数据框的前n行
df.tail(3) # 查看数据框的最后n行
df.shape # 查看数据框的行数与列数
df.info() # 查看数据框 (DataFrame) 的索引、数据类型及内存信息
df.describe() # 对于数据类型为数值型的列，查询其描述性统计的内容
s = pd.Series([1,2,3,3,4,np.nan,5,5,5,6,7])
s.value_counts(dropna=False) # 查询每个独特数据值出现次数统计
df = pd.DataFrame(np.random.rand(5,5),columns=list('ABCDE'))
df['C']  # 以数组 Series 的形式返回选取的列
df[['B','E']]
s = pd.Series(np.array(['I','Love','Data']))
s.iloc[0]# 按照位置选取
s.index=['index_one','index_two','index_three']
s.loc['index_one'] # 按照索引选取
df = pd.DataFrame(np.random.rand(5,5),columns=list('ABCDE'))
df.iloc[0,:] # 选取第一行
df.iloc[0,0] # 选取第一行的第一个元素
df = pd.DataFrame({'A':np.array([1,np.nan,2,3,6,np.nan]),
                 'B':np.array([np.nan,4,np.nan,5,9,np.nan]),
                  'C':'foo'})
df.columns = ['a','b','c'] # 重命名数据框的列名称
df = pd.DataFrame({'A':np.array([1,np.nan,2,3,6,np.nan]),
                 'B':np.array([np.nan,4,np.nan,5,9,np.nan]),
                  'C':'foo'})
pd.isnull() # 检查数据中空值出现的情况，并返回一个由布尔值(True,Fale)组成的列
pd.notnull() # 检查数据中非空值出现的情况，并返回一个由布尔值(True,False)组成的列
df.dropna()
df.dropna(axis=1) # 移除数据框 DataFrame 中包含空值的列
test = df.dropna(axis=1,thresh=1)
df.fillna('Test') # 将数据框 DataFrame 中的所有空值替换为 x
s = pd.Series([1,3,5,np.nan,7,9,9])
s.fillna(s.mean())#将所有空值替换为平均值
s.astype(float) # 将数组(Series)的格式转化为浮点数
s.replace(1,'one') # 将数组(Series)中的所有1替换为'one'
s.replace([1,3],['one','three']) # 将数组(Series)中所有的1替换为'one', 所有的3替换为'three'
df = pd.DataFrame(np.random.rand(4,4))
df.rename(columns=lambda x: x+ 2)# 将全体列重命名
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.rename(columns={'old_name': 'new_ name'}) # 将选择的列重命名
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.set_index('B')# 改变索引
df = pd.DataFrame(np.random.rand(10,5))
df.rename(index = lambda x: x+ 1)# 改变全体索引
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df[df['A'] > 0.5]# 选取数据框df中对应行的数值大于0.5的全部列
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.sort_values('E')# 按照数据框的列col1升序(ascending)的方式对数据框df做排序
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.sort_values('A',ascending=False)# 按照数据框的列col2降序(descending)的方式对数据框df做排序
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.sort_values(['A','E'],ascending=[True,False])# 按照数据框的列col1升序，col2降序的方式对数据框df做排序
df = pd.DataFrame({'A':np.array(['foo','foo','foo','foo','bar','bar']),
      'B':np.array(['one','one','two','two','three','three']),
     'C':np.array(['small','medium','large','large','small','small']),
     'D':np.array([1,2,2,3,3,5])})

df.groupby('A').count()# 按照某列对数据框df做分组
df = pd.DataFrame({'A':np.array(['foo','foo','foo','foo','bar','bar']),
      'B':np.array(['one','one','two','two','three','three']),
     'C':np.array(['small','medium','large','large','small','small']),
     'D':np.array([1,2,2,3,3,5])})

df.groupby(['B','C']).sum()# 按照列col1和col2对数据框df做分组
df.groupby('B')['D'].mean()# 按照列col1对数据框df做分组处理后，返回对应的col2的平均值
df.pivot_table(df,index=['A','B'],
               columns=['C'],aggfunc=np.sum) # 做透视表，索引为col1,针对的数值列为col2和col3，分组函数为平均值
df.groupby('A').agg(np.mean)
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.apply(np.mean) # 对数据框df的每一列求平均值
df.apply(np.max,axis=1) # 对数据框df的每一行求最大值
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df1.append(df2)# 在数据框df2的末尾添加数据框df1，其中df1和df2的列数应该相等
pd.concat([df1, df2],axis=1) # 在数据框df1的列最后添加数据框df2,其中df1和df2的行数应该相等
df1.join(df2,on='key',how='inner') # 对数据框df1和df2做内连接，其中连接的列为col1
df = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))
df.describe() # 得到数据框df每一列的描述性统计
df.mean() # 得到数据框df中每一列的平均值
df.corr() # 得到数据框df中每一列与其他列的相关系数
df.count() # 得到数据框df中每一列的非空值个数
df.max() # 得到数据框df中每一列的最大值
df.min() # 得到数据框df中每一列的最小值
df.median() # 得到数据框df中每一列的中位数
df.std() # 得到数据框df中每一列的标准差
