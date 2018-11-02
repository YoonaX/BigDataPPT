# -*- coding: utf-8 -*-
import numpy as np
from numpy.random.mtrand import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
# 处理数据
filename = '../data/4.4.3-wine.csv'
data = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
shuffle(data)
X = data[:, :-1]
y = data[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=99)

# 训练模型
clf = RandomForestClassifier(n_estimators=3)
#各种参数的含义和设置！
clf.fit(x_train, y_train)
label_predict = clf.predict(x_test)
# print(clf.estimators_[0].tree_.n_node_samples)
#模型评估
from sklearn.metrics import classification_report
print(classification_report(y_test, label_predict))

for i in range(len(clf.estimators_)):
    dot_data = tree.export_graphviz(clf.estimators_[i], out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = graphviz.Source(dot_data) 
    graph.render('CLF%d_RF.dot' % (i + 1), "img/", view=True)