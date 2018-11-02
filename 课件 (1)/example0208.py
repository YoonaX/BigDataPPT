# coding: utf-8
import matplotlib.pyplot as plt
data=[-35,10,21,30,40,50,60,71,126]
flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}
# plt.grid(True, linestyle = "-.", color = "black") 
plt.boxplot(data,notch=False,flierprops=flierprops)
plt.show()