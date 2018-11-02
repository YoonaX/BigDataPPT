# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

sort_list = []
List = []
for i in range(200):
    x = random.randint(0, 100)
    List.append(x)
    sort_list.append(x)
print(List)


for i in range(len(sort_list) - 1):
    for j in range(i + 1, 0, -1):
        if sort_list[j] < sort_list[j - 1]:
            temp = sort_list[j]
            sort_list[j] = sort_list[j - 1]
            sort_list[j - 1] = temp
print(sort_list)

average_list = []
middle_list = []
for i in range(8):
    average = 0
    for j in range(25):
        average += sort_list[i * 25 + j]
    average /= 25
    average_list.append(average)
    middle_list.append(sort_list[i * 25 + 12])
print(average_list)
print(middle_list)

average_smooth_list = []
middle_smooth_list = []
for i in range(len(List)):
    sort_list_index = sort_list.index(List[i])
    average_list_index = int(sort_list_index / 25)
    average_smooth_list.append(average_list[average_list_index])
    middle_smooth_list.append(middle_list[average_list_index])
print(average_smooth_list)

x = np.array([])
for i in range(200):
    x = np.append(x, i)

y = np.array([])
for i in range(200):
    y = np.append(y, List[i])

line1, = plt.plot(x, y, color='r')
# plt.subplot(211)
# plt.show()
# plt.close()

y = np.array([])
for i in range(200):
    y = np.append(y, average_smooth_list[i])

line2, = plt.plot(x, y, color='y')

y = np.array([])
for i in range(200):
    y = np.append(y, middle_smooth_list[i])

line3, = plt.plot(x, y, color='b')

# plt.subplot(211)
plt.legend([line1, line2, line3], ["Before", "Average-Smooth", "Middle-Smooth"], loc=1)
plt.show(fontproperties="SimHei")