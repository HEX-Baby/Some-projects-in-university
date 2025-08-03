import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 给定的 x 和 y 数据点
T = [700, 720, 740, 760, 780]
V = [0.0977, 0.1218, 0.1406, 0.1551, 0.1664]

# 生成三次样条插值函数
cs = CubicSpline(T, V, bc_type='natural')  # 'natural' 代表自然边界条件

x_new = np.linspace(700, 800, 6)
y_new = cs(x_new)

plt.scatter(T, V, color='red', label="原始数据点")
plt.plot(x_new, y_new, label="三次样条插值曲线")
plt.legend()
plt.title("Cubic Spline Interpolation")
plt.show()




# 生成线性插值函数
linear_interp = interp1d(T, V, kind='linear')


x_new1 = np.linspace(700, 780, 6)
y_new1 = linear_interp(x_new1)

# 绘制原始数据点和插值曲线
plt.scatter(T, V, color='red', label="原始数据点")
plt.plot(x_new1, y_new1, label="线性插值曲线", linestyle="dashed")
plt.legend()
plt.title("Linear Interpolation")
plt.show()



print(f"三次样条插值：T=750和T=770时的数据：{cs(750)},{cs(770)}")
print(f"线性插值：T=750和T=770时的数据：{linear_interp(750)},{linear_interp(770)}")