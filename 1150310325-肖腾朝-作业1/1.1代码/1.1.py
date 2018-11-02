# -*- coding: utf-8 -*-
import random
while True:
    print("Please input K(0 <= K <= 100):")
    K = input()
    X = int(K)
    print("输入的K值为：{}".format(X))
    if (X >= 0) and (X <= 100):
        break

file = open("1.1.txt", 'r')
List = []
for line in file:
    # print(line)
    line_list = line.split(",")
    for i in line_list:
        int_i = int(i)
        List.append(int_i)
print("排序前的列表为：{}".format(List))
file.close()

for i in range(len(List) - 1):
    for j in range(i + 1, 0, -1):
        if List[j] < List[j - 1]:
            Temp = List[j]
            List[j] = List[j - 1]
            List[j - 1] = Temp
        else:
            break

print("排序后的列表为：{}".format(List))

Pos = 1 + (len(List) - 1) * X / 100
print("算得的Pos值为：{}".format(Pos))
int_Pos = int(Pos)
if Pos - int_Pos == 0:
    print("{} 百分位数计算结果为(Pos为整数，线性插值、下界、上界、中点和相同)：{}".format(X, List[int_Pos - 1]))
else:
    print("{} 百分位数计算结果为(线性插值)：{}".format(X, (List[int_Pos - 1] + (List[int_Pos] - List[int_Pos - 1]) * X / 100 )))
    print("{} 百分位数计算结果为(下界)：{}".format(X, List[int_Pos - 1]))
    print("{} 百分位数计算结果为(上界)：{}".format(X, List[int_Pos]))
    print("{} 百分位数计算结果为(中点)：{}".format(X, (List[int_Pos - 1] + List[int_Pos]) / 2))
    if (Pos - int_Pos) == 0.5:
        Judge = random.randomint(0,1)
        print("{} 百分位数计算结果为（最近邻）：{}".format(X, List[int_Pos - 1 + Judge]))
    else:
        exp = lambda x: 1 if x > 0.5 else 0
        b = exp(Pos - int_Pos)
        print("{} 百分位数计算结果为（最近邻结果）：{}".format(X, List[int_Pos - 1 + b]) )