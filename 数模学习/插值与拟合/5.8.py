import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import BarycentricInterpolator
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#原始数据
x = np.linspace(-2,4.9,24)

y = np.array([0.1029, 0.1174, 0.1316, 0.1448, 0.1556, 0.1662, 0.1733, 0.1775,
              0.1785, 0.1764, 0.1711, 0.1630, 0.1526, 0.1402, 0.1266, 0.1122,
              0.0977, 0.0835, 0.0702, 0.0588, 0.0479, 0.0373, 0.0291, 0.0224])


#三次样条插值
cs = CubicSpline(x, y, bc_type='natural')  # 'natural' 代表自然边界条件

x_new_cs = np.linspace(-2, 5, 50)
y_new_cs = cs(x_new_cs)

# 绘制原始数据点和插值曲线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new_cs, y_new_cs, label="三次样条插值曲线")
plt.legend()
plt.title("Cubic Spline Interpolation")
plt.show()


#拉格朗日插值


# 给定的插值点 (x, y)

# 生成拉格朗日插值多项式
poly = lagrange(x, y)

# 生成插值曲线
x_new = np.linspace(min(x), max(x), 100)
y_new = poly(x_new)

# 绘制结果
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new, label="拉格朗日插值曲线")
plt.legend()
plt.title("Lagrange Interpolation")
plt.show()

# 打印多项式表达式
print("拉格朗日插值多项式:")
print(poly)

#牛顿插值

# 创建插值对象
newton_interp = BarycentricInterpolator(x, y)

# 计算新数据点
y_new_newton = newton_interp(x_new)

# 绘图
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new_newton, label="Newton 插值 (Scipy)", linestyle="dashed")
plt.legend()
plt.show()


#计算均方误差

y_true = y

y_pred_cs = cs(x)
y_pred_lagrange = poly(x)
y_pred_newton = newton_interp(x)

mes_cs = mean_squared_error(y_true=y_true,y_pred=y_pred_cs)
mes_lagrange = mean_squared_error(y_true=y_true,y_pred=y_pred_lagrange)
mes_newton = mean_squared_error(y_true=y_true,y_pred=y_pred_newton)
print(f"三次样条插值的均方误差MSE = {mes_cs}")
print(f"Lagrange插值的均方误差MSE = {mes_cs}")
print(f"Newton插值的均方误差MSE = {mes_cs}")