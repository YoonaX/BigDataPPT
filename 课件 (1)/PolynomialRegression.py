# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def f(x):
    """ 使用多项式插值拟合的函数"""
    return x * np.sin(x)

# 为了画函数曲线生成的数据
x_plot = np.linspace(0, 10, 100)
# 生成训练集
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(9)
rng.shuffle(x)
x = np.sort(x[:30])
y = f(x)
# 转换为矩阵形式,np.newaxis 为 numpy.ndarray（多维数组）增加一个轴.np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名。
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
colors = ['teal', 'yellowgreen', 'gold']
linestyles = ["-", "--", ":"]
fig = plt.figure()
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=1,label="ground truth")
plt.scatter(x, y, color='navy', s=10, marker='o', label="training points")
for count, degree in enumerate([3, 4, 5]):
    Poly = PolynomialFeatures(degree)
    X_poly = Poly.fit_transform(X)
    model = Ridge()
    model.fit(X_poly, y)
    y_plot = model.predict(Poly.fit_transform(X_plot))
    plt.plot(x_plot, y_plot, color=colors[count], linestyle=linestyles[count], linewidth=2,
             label="degree %d" % degree)
    plt.legend(loc='lower left')
plt.show()
