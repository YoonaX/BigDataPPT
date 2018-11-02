# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)
x = np.random.randint(0, 100, 50)
y1 = 0.8*x + np.random.normal(0, 15, 50)
y2 = 100 - 0.7*x + np.random.normal(0, 15, 50)
y3 = np.random.randint(0, 100, 50)
r1=np.corrcoef(x, y1)
r2=np.corrcoef(x, y2)
r3=np.corrcoef(x, y3)

plt.subplot(131)
plt.scatter(x, y1,color='k')
plt.subplot(132)
plt.scatter(x, y2,color='k')
plt.subplot(133)
plt.scatter(x, y3,color='k')
print (r1[0][1])
print (r2[0][1])
print (r3[0][1])
plt.show()