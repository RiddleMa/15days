from numpy import *
a = arange(15).reshape(3, 5)
a.shape#数组的维度
ndarray.ndim#数组轴的个数
ndarray.size# 数组元素的总个数
ndarray.dtype#数组中元素类型的对象
ndarray.itemsize#每个元素的字节大小
ndarray.data#实际数组元素的缓冲区，通常我们不需要使用这个属性
type(a)#ndarray
a = array( [2,3,4] )#创建数组
#数组将序列包含序列转化成二维的数组，序列包含序列包含序列转化成三维数组等等。
b = array([(1.5,2,3),(4,5,6)])
c = array([[1,2],[3,4]], dtype=complex )#数组类型可以在创建时显示指定 复数
zeros((3,4))
ones((2,3,4),dtype=int16)
empty((2,3))#创建一个内容随机并且依赖与内存状态的数组。默认创建的数组类型(dtype)都是float64。
arange(10,30,5)#返回数组而不是列表

a = array([20,30,40,50])
b = arange(4)
c = a-b  #array([20, 29, 38, 47])
a<35  #array([ True,  True, False, False], dtype=bool)
a*b #array([  0,  30,  80, 150]) 按照元素计算，并非矩阵乘法
dot(a,b.reshape(4,1))#array([260])

b = arange(12).reshape(3,4)
b.sum(axis=0)                            # sum of each column
b.min(axis=1)                            # min of each row
b.cumsum(axis=1)                         # cumulative sum along each row各行累加

#Numpy提供常见的数学函数如 sin, cos 和 exp
'''all, alltrue, any, apply along axis, argmax, argmin, argsort, average,     
bincount, ceil, clip, conj, conjugate, corrcoef, cov, cross, cumprod, cumsum, 
diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, 
nonzero, outer, prod, re, round, sometrue, sort, std, sum, trace, transpose, 
var, vdot, vectorize, where'''
B = arange(3)
exp(B) # e的幂
sqrt(B) #开方
a[:6:2] = -1000 # 类似a[0:6:2] = -1000   从0开始，每隔2个赋值，直到6
a[ : :-1]       #倒序

a = floor(10*random.random((3,4)))
a.ravel()#展平
a.shape = (6, 2)
a.transpose()#输出转置，不该变原数组

#堆叠
a = floor(10*random.random((2,2)))
b = floor(10*random.random((2,2)))
vstack((a,b))#纵向堆叠
hstack((a,b))#横向堆叠
column_stack((a,b)) #以列将一维数组合成二维数组，它等同与vstack对一维数组
a=array([4.,2.])
b=array([2.,8.])
a[:,newaxis]  # This allows to have a 2D columns vector
column_stack((a[:,newaxis],b[:,newaxis]))
vstack((a[:,newaxis],b[:,newaxis])) # The behavior of vstack is different
r_[1:4,0,4]
#分割
a = floor(10*random.random((2,12)))
hsplit(a,3)   # Split a into 3
hsplit(a,(3,4))   # Split a after the third and the fourth column
#vsplit沿着纵向的轴分割，array split允许指定沿哪个轴分割。
#复制
# 完全不拷贝
a = arange(12)
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
b.shape = 3,4    # changes the shape of a
a.shape
#视图
c = a.view()
c is a#False
c.base is a  # c is a view of the data owned by a
c.flags.owndata #False
c.shape = 2,6                      # a's shape doesn't change
a.shape
c[0,4] = 1234                      # a's data changes a
a
s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
a
#深复制 完全复制数组和它的数据
d = a.copy()                          # a new array object with new data is created
d is a #False
d.base is a                           # d doesn't share anything with a


