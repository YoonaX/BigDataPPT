# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split


def show(table, predict, index, xlabel, ylabel):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel(xlabel, fontproperties="SimHei")
    plt.ylabel(ylabel, fontproperties="SimHei")
    ax1.scatter(table[:, index], predict)
    plt.savefig('D:/pythonprogram/BigData/picture/{}.png'.format(xlabel))
    plt.show()
    plt.close()


data = np.loadtxt("D:\本科数据挖掘\课件 (1)\housing.data")
table, predict = data[:, :-1], data[:, -1]
# table_train, table_test, predict_train, predict_test = train_test_split(table, predict, test_size=0.2, random_state=42)

table_train, table_test, predict_train, predict_test = train_test_split(table, predict, test_size=0.2)
print(table_train)
table_train = table[:480]
table_test = table[480: 600]
predict_train = predict[:480]
predict_test = predict[480: 600]

print("-" * 50)
print(table_train)

train = np.column_stack((table_train, predict_train))
np.savetxt('D:\\pythonprogram\\BigData\\train.csv', train, delimiter=' ')
test = np.column_stack((table_test, predict_test))
np.savetxt('D:\\pythonprogram\\BigData\\test.csv', test, delimiter=' ')

# show(table, predict, 0, u'CRIM(城镇人均犯罪率)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 1, u'ZN(占住宅用地面积逾25,000平方英尺的比例)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 2, u'INDUS(每镇非零售经营面积的比例)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 3, u'CHAS(Charles River哑变量(= 1，若为径流界河;0否则))', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 4, u'NOX(一氧化氮浓度(千万分之一))', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 5, u'RM(每个住宅的平均房间数)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 6, u'AGE(1940年以前建造的业主自住单位的比例)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 7, u'DIS(加权距离波士顿五个就业中心)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 8, u'RAD(径向公路可达性指数)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 9, u'TAX(每1万美元的全额财产税税率)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 10, u'PTRATIO(学生与教师的比率按城市而定)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 11, u'B(由城镇黑人的比例)', u'MEDV(业主自住房屋价值中位数为1000美元)')
# show(table, predict, 12, u'LSTAT(人口地位较低)', u'MEDV(业主自住房屋价值中位数为1000美元)')


line = LinearRegression()
ridge = Ridge()

line.fit(table_train, predict_train)
ridge.fit(table_train, predict_train)

line_y_pre = line.predict(table_test)
ridge_y_pre = ridge.predict(table_test)

plt.plot(predict_test, label='True')
plt.plot(line_y_pre, label='Line', color='r')
plt.plot(ridge_y_pre, label='Ridge', color='y')
plt.legend()
plt.savefig('D:/pythonprogram/BigData/picture/回归图.png')
plt.show()