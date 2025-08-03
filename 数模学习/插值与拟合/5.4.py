import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.array([36.9, 45.7, 63.7, 77.8, 84.0, 87.5])
y = np.array([181, 197, 235, 270, 283, 292])

# 使用 np.polyfit 进行线性最小二乘拟合
a, b = np.polyfit(x, y, 1)  # 1 表示拟合 1 次多项式，即 y = ax + b

x_new = np.linspace(35, 90, 100)
y_new = a * x_new + b

# 绘制数据点和拟合直线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new, label=f"拟合直线: y = {a:.2f}x + {b:.2f}", linestyle="dashed")
plt.legend()
plt.title("最小二乘法拟合直线")
plt.show()

print(f"拟合直线方程: y = {a:.2f}x + {b:.2f}")