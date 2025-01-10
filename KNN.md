KNN是一种基础的机器学习算法，属于监督学习类别。它的核心思想是：物以类聚，通过分析新数据点周围最近的K个样本的类别来预测该数据点的类别

同时k近邻算法是非常特殊的，可以被认为是没有模型的算法。为了和其他算法统一，可以认为训练数据集就是模型本身
## 自定义模型
预测良性肿瘤和恶性肿瘤模型，数据集X和特征集Y，
```python
# X的数据为5行2列的矩阵
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[6.745401,6.507143],[7.319939,5.986585],[1.560186,1.559945],[0.580836,3.661761],[1.011150,4.080726]])

# 特征集Y是5行1列的矩阵
Y = np.array([0,0,1,1,1])

plt.scatter(X[Y==0,0], X[Y==0,1], color='g')
plt.scatter(X[Y==1,0], X[Y==1,1], color='r')
plt.show()
```

![knn](https://github.com/arqady01/machine-learning/blob/main/img/knn.png)

接着不妨事先有个大概，比如一个新的点是(2,3)

```python
x = np.array([2,3]) # 待预测的肿瘤
plt.scatter(x[0],x[1],color='b') # 在图中的位置
```

![距离](https://github.com/arqady01/machine-learning/blob/main/img/knn2.png)

第二步，计算距离

$$平面之间两点的间距公式：d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

接着要计算待预测的点A距离其他5个点的间距分别是多少
1. 点1的横坐标 - 点A的横坐标
2. 点1的纵坐标 - 点A的纵坐标
3. 平方相加
4. 开根号
5. 重复，得到一个矩阵
用numpy的方式就可以简化成先把数据集X 减去 点A（矩阵相减），在平方开方运算：

```python
from math import sqrt
distances = [] # 定义一个空列表
for x_train in X: # x_train是一个1行2列的矩阵
    temp = x_train - x # 同形状的矩阵相减，得到横坐标和纵坐标的距离矩阵
    temp_square = temp ** 2 # 将距离矩阵平方
    d_square = np.sum(temp_square) # 求和，得到两点间距的平方和
    d = sqrt(d_square) # 开方，得到两点的距离
    distances.append(d)
```

最后得到distances矩阵：[5.900752720903495, 6.100937708741665, 1.5057206771579514, 1.56587166779944, 1.4648525555754752]

最后业务处理，我们得到了distances即每个点距离待处理点的间距数值，接下来排个序

```python
nearest = np.argsort(distances)
# 输出：[4 2 3 0 1]

k = 3 # 业务需求，即距离待处理点最近的3个点
topK_y = [Y[i] for i in nearest[:k]] # 得到距离待处理点最近的3个点是不是肿瘤
```

对于人类来说很好观察，但是对于机器不知道这三个点中，是否癌症的票数多少

```python
from collections import Counter
votes = Counter(topK_y) # 得到Counter({0:0, 1:3})
votes.most_common(1) # 得到票数最高的一个
```

## 使用sklearn的KNN

```python
from sklearn.neighbors import KNeighborsClassifier
kNN_classifier = KNeighborsClassifier(n_neighbors=3) # 用默认构造来创建一个对象
kNN_classifier.fit(X, Y) # 拟合
kNN_classifier.predict(x) # 预测待处理点x
```

> n_neighbors 表示使用的邻居数量，默认为5

## 划分数据集

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris() # 加载鸢尾花的数据
X = iris.data # 数据集
y = iris.target # 特征集
X.shape # 输出：(150, 4)，即150行4列
y.shape # 输出：(150,)，即150行1列

# 数据拆分：将数据集的80%用于训练20%用于验证，防止数据排好序，所以随机化

# 又因为X和Y是对应的，所以不能对X随机化后再对y随机化

# 解决方案：将所有数据的编号随机化

shuffle_index = np.random.permutation(len(X)) # 生成随机化序列排列
test_size = int(len(X) * 0.2) # 验证（测试）数据集的大小为数据集的20%个
test_index = shuffle_index[:test_size] # 将数据的前20%用于验证
train_index = shuffle_index[test_size:] # 将数据的后80%用于训练

# 构建训练集和测试集
X_train = X[train_index]
y_train = y[train_index]
X_test = X[test_index]
y_test = y[test_index]
```

封装成函数形式：

```python
def train_test_split(X, y, test_ratio=0.2, seed=None):
    # 将数据X和y按照test_radio分割成X_train，X_test，y_train，y_test
    if seed:
        np.random.seed(seed)
    shuffled_index = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
```

调用api形式

```python
X_train, X_test, y_train, y_test = sklearn.model_selection(iris.data, iris.target, test_size=0.2, random_state=22)
```

## 数据归一化

主要消除量纲的影响，比如肿瘤受肿瘤块和发现天数影响，而天数的量级会比大小影响大很多。

解决办法：将所有数据映射到统一尺度

- 最值归一化：把所有数据映射到0-1之间

$$ x_{scale} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

适用于有明显边界的情况，比如考试分数，如果没有明显边界比如全国人民收入，就不适合最值归一化

```python
# 生成一个从0到99的包含100个随机整数的一维数组
x = np.random.randomint(0, 100, size=100)
# (矩阵 - 数值) / (数值 - 数值) 得到一个矩阵
(x - np.min(x)) / (np.max(x) - np.min(x))
```

最终的结果都在0-1之间，完成了最值归一化

- 均值方差归一化：把所有数据归一到均值为0方差为1的分布中

$$ x_{scale} = \frac{x - x_{mean}}{s} $$

