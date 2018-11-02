# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import precision_recall_curve  # 准确率与召回率
import numpy as np
import os
import pydotplus
from sklearn import metrics
# Do classification task,
# then get the ground truth and the predict label named y_true and y_pred

os.environ["PATH"] += os.pathsep + "E:\\Grapvz2.38\\bin"


def get_data():
    file_path = "D:\\本科数据挖掘\\课件 (1)\\iris.txt"
    data = np.loadtxt(file_path, delimiter=",", dtype=str)
    # print(data)
    iris_x = data[:, :-1]  # 此处有冒号，不显示最后一列
    iris_y = data[:, -1]  # 此处没有冒号，直接定位

    x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.2)
    # print(x_train)
    # print(y_test)

    return iris_x, iris_y, x_train, x_test, y_train, y_test


def train(iris_x=None, iris_y=None, x_train=None, x_test=None, y_train=None, y_test=None, flag=1):
    if flag == 0:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(iris_x, iris_y)
        feature = ['spal_length', 'speal_width', 'petal_length', 'petal_width']
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature, filled=True, rounded=True,
                                        special_characters=True, precision=4)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("workiris_all.pdf")
        return  None

    else:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train, y_train)
        predict_data = clf.predict(x_test)
        predict_proba = clf.predict_proba(x_test)
        # print("predict_proba")
        # print(predict_proba)

        feature = ['spal_length', 'speal_width', 'petal_length', 'petal_width']
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature, filled=True, rounded=True,
                                        special_characters=True, precision=4)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("workiris_test.pdf")

        classify_report = metrics.classification_report(y_test, clf.predict(x_test))
        print("classify_report")
        print(classify_report)

        confusion_matrix = metrics.confusion_matrix(y_train, clf.predict(x_train))
        print("confusion_matrix")
        print(confusion_matrix)

        # overall_accuracy = metrics.accuracy_score(y_train, clf.predict(x_train))
        # print("overall_accuracy")
        # print(overall_accuracy)
        #
        # acc_for_each_class = metrics.precision_score(y_train, clf.predict(x_train), average=None)
        # print("acc_for_each_class")
        # print(acc_for_each_class)
        #
        # overall_accuracy = np.mean(acc_for_each_class)
        # print("overall_accuracy")
        # print(overall_accuracy)

        test_accuracy = metrics.accuracy_score(y_test, clf.predict(x_test))
        print("test_accuracy")
        print(test_accuracy)

        test_acc_for_each_class = metrics.precision_score(y_test, clf.predict(x_test), average=None)
        print("test_acc_for_each_class")
        print(test_acc_for_each_class)

        test_overall_accuracy = np.mean(test_acc_for_each_class)
        print("test_overall_accuracy")
        print(test_overall_accuracy)

        return classify_report


if __name__ == "__main__":
    iris_x, iris_y, x_train, x_test, y_train, y_test = get_data()
    train(iris_x=iris_x, iris_y=iris_y, flag=0)
    train(x_train=x_train, y_train=y_train,  x_test=x_test, y_test=y_test, flag=1)
