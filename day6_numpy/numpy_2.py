from numpy import *

a = arange(12)**2                          # the first 12 square numbers
i = array( [ 1,1,3,8,5 ] )                 # an array of indices
a[i]
j = array( [ [ 3, 4], [ 9, 7 ] ] )         # a bidimensional array of indices
a[j]                                       # the same shape as j
#当被索引数组a是多维的时，每一个唯一的索引数列指向a的第一维5。以下示例通过将图片标签用调色版转换成色彩图像展示了这种行为。
palette = array( [ [0,0,0],                # 黑色
                   [255,0,0],              # 红色
                   [0,255,0],              # 绿色
                   [0,0,255],              # 蓝色
                   [255,255,255] ] )       # 白色
image = array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                 [ 0, 3, 4, 0 ]  ] )
palette[image]                            # the (2,4,3) color image
#多维索引
a = arange(12).reshape(3,4)
i = array( [ [0,1],                        # indices for the first dim of a
             [1,2] ] )
j = array( [ [2,1],                        # indices for the second dim
             [3,3] ] )
a[i,j]                                     # i and j must have equal shape
a[i,2]
a[:,j]                                     # i.e., a[ : , j]
l = [i,j]
a[l]                                       # 与 a[i,j] 相等

#另一个常用的数组索引用法是搜索时间序列最大值
time = linspace(20, 145, 5)                 # time scale
data = sin(arange(20)).reshape(5,4)         # 4 time-dependent series
time
ind = data.argmax(axis=0)                   # index of the maxima for each series 每一列最大值序号
ind
time_max = time[ ind]                       # times corresponding to the maxima 根据之前的序号排列时间序列
data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...
time_max
all(data_max == data.max(axis=0))   # True
a = arange(5)       #array([0, 1, 2, 3, 4])
a[[1,3,4]] = 0      #array([0, 0, 2, 0, 0])
#通过布尔数组索引
a = arange(12).reshape(3,4)
b = a > 4
b       # b is a boolean with a's shape
a[b]    # 1d array with the selected elements
a[b] = 0    # All elements of 'a' higher than 4 become 0

a = arange(12).reshape(3,4)
b1 = array([False,True,True])             # first dim selection
b2 = array([True,False,True,False])       # second dim selection
a[b1,:]                                   # selecting rows
a[b1]                                     # same thing
a[:,b2]                                   # selecting columns
a[b1,b2]                                  # a weird thing to do
a = array([2,3,4,5])
b = array([8,5,4])
c = array([5,4,6,8,3])
ax,bx,cx = ix_(a,b,c)
ax.shape, bx.shape, cx.shape        #((4, 1, 1), (1, 3, 1), (1, 1, 5))
result = ax+bx*cx
# 可以实行如下简化：
def ufunc_reduce(ufct, *vectors):
    vs = ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r
ufunc_reduce(add,a,b,c)

#线性代数
# 简单数组运算
from numpy import *
from numpy.linalg import *
a = array([[1.0, 2.0], [3.0, 4.0]])
print (a)
a.transpose()
inv(a)#逆
u = eye(2) # unit 2x2 matrix; "eye" represents "I"单位矩阵
j = array([[0.0, -1.0], [1.0, 0.0]])
dot (j, j) # matrix product
trace(u)  # trace 迹 主对角线之和
y = array([[5.], [7.]])
solve(a, y)  #a*x=y x?
eig(j) #返回包含特征值和对应特征向量的元组
# 矩阵类
A = matrix('1.0 2.0; 3.0 4.0')
type(A)  # file where class is defined numpy.matrixlib.defmatrix.matrix
A.T  # transpose
X = matrix('5.0 7.0')
Y = X.T
print (A*Y)  # matrix multiplication
print (A.I)  # inverse
solve(A, Y)  # solving linear equation
# Numpy中数组和矩阵有些重要的区别。
# 矩阵是继承自Numpy数组对象的二维数组对象。
A = arange(12)
A.shape = (3,4)
M = mat(A.copy())
print (type(A),"  ",type(M))#<class 'numpy.ndarray'>    <class 'numpy.matrixlib.defmatrix.matrix'>
print (A)
print (M)
print (A[:])
print (A[:].shape)
print (M[:])
print (M[:].shape)#相同
print (A[:,1])#产生一维数组
print (A[:,1].shape)
print (M[:,1]); #产生二维数组
print (M[:,1].shape)
# 假如我们想要一个数组的第一列和第三列，一种方法是使用列表切片：
A[:,[1,3]]
# 稍微复杂点的方法是使用 take() 方法 method:
A[:,].take([1,3],axis=1)
# 或者我们仅仅使用A[1:,[1,3]]。还有一种方法是通过矩阵向量积(叉积)。
A[ix_((1,2),(1,3))]
# 我们想要保留第一行大于1的列。一种方法是创建布尔索引：
A[0,:]>1            #array([False, False,  True,  True], dtype=bool)
A[:,A[0,:]>1]
# 但是索引矩阵没这么方便。
M[0,:]>1            #matrix([[False, False,  True,  True]], dtype=bool)
M[:,M.A[0,:]>1]     #使用M.A转换成数组
# 如果我们想要在矩阵两个方向有条件地切片，我们必须稍微调整策略
A[A[:,0]>2,A[0,:]>1]
M[M.A[:,0]>2,M.A[0,:]>1]
A[ix_(A[:,0]>2,A[0,:]>1)]
M[ix_(M.A[:,0]>2,M.A[0,:]>1)]
#小技巧
#使用-1缺省
a = arange(30)
a.shape = 2,-1,3  # -1 means "whatever is needed"
a.shape
# 向量组合(stacking)   column_stack 、dstack、hstack 和 vstack
x = arange(0,10,2)
y = arange(5)
m = vstack([x,y])
xy = hstack([x,y])

# 直方图(histogram)
import numpy
import pylab
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = numpy.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
pylab.hist(v, bins=50, normed=1)
pylab.title('Matplotlib Version')# matplotlib version (plot)
pylab.show()
# Compute the histogram with numpy and then plot it
(n, bins) = numpy.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.title('Numpy Version')
pylab.show()