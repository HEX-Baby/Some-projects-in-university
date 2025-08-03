
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
'''
使用最小二乘法拟合最佳次数
'''
# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#原始数据
x = np.linspace(-2,4.9,24)

y = np.array([0.1029, 0.1174, 0.1316, 0.1448, 0.1556, 0.1662, 0.1733, 0.1775,
              0.1785, 0.1764, 0.1711, 0.1630, 0.1526, 0.1402, 0.1266, 0.1122,
              0.0977, 0.0835, 0.0702, 0.0588, 0.0479, 0.0373, 0.0291, 0.0224])

a = []
error = []


for i in range(1,10):
    coffes = np.polyfit(x, y, i)
    a.append(coffes)
    y_pred = np.polyval(coffes,x)
    MSE = mean_squared_error(y_true=y, y_pred=y_pred)

    p = i +1
    RSD = np.sqrt(MSE * (len(y) / (len(y) - p)))
    error.append(RSD)

for i in range(1,10):
    print(f"次数为{i}:")
    print(a[i - 1])
    print(f'剩余标准差：{error[i - 1]}')