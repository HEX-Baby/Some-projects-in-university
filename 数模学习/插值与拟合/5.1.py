import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 给定的 x 和 y 数据点
x = np.linspace(0,10,1000)
def g_x(x):
    return (3 * (x ** 2) + 4 * x + 6) * np.sin(x) / (x ** 2 + 8 * x + 6)

y = g_x(x)



# 生成三次样条插值函数
cs = CubicSpline(x, y, bc_type='natural')  # 'natural' 代表自然边界条件

# 在更细的 x 轴上计算插值
x_new = np.linspace(0, 10, 100)
y_new = cs(x_new)

# 绘制原始数据点和插值曲线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new, label="三次样条插值曲线")
plt.legend()
plt.title("Cubic Spline Interpolation")
plt.show()

print(cs.c)
#计算定积分

a = 0
b = 10

integral_value = cs.integrate(a, b)

print(f"∫ S(x) dx 从 {a} 到 {b} 的积分值：{integral_value:.6f}")



# 计算不定积分 S(x) 的原函数
cs_integral = cs.antiderivative()

# 计算 S(x) 在新的 x 值上的积分值
y_integral = cs_integral(x_new)

# 绘制原始函数和积分曲线
plt.plot(x_new, cs(x_new), label="三次样条插值")
plt.plot(x_new, y_integral, label="不定积分曲线", linestyle="dashed")
plt.legend()
plt.title("Cubic Spline Interpolation and Its Integral")
plt.show()


###
'''
计算给定函数定积分
'''
from scipy.integrate import quad

result, error = quad(g_x, a, b)
print(f"∫ f(x) dx 从 {a} 到 {b} 的积分值：{result:.6f}")
print(f"数值误差估计：{error:.6e}")