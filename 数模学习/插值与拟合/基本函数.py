'''
s三次样条插值
'''
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
########################################################求出表达式#################
"""
cs.c[0, i] 是 i 号区间的 三次项系数 a

 
cs.c[1, i] 是 i 号区间的 二次项系数 b

cs.c[2, i] 是 i 号区间的 一次项系数 c
 
cs.c[3, i] 是 i 号区间的 常数项 d

cs.x[i] 表示第 i 个区间的起点
"""
cs.c  # 形状为 (4, n-1) 的矩阵



'''
线性插值函数
scipy.interpolate.interp1d() 创建一个线性插值函数，可以用来计算新的插值点。
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 给定的 x 和 y 数据点
x = np.array([0, 2, 4, 6, 8, 10])
y = (3 * (x ** 2) + 4 * x + 6) * np.sin(x) / (x ** 2 + 8 * x + 6)

# 生成线性插值函数
linear_interp = interp1d(x, y, kind='linear')

# 在更细的 x 轴上计算插值
x_new = np.linspace(0, 10, 100)
y_new = linear_interp(x_new)

# 绘制原始数据点和插值曲线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new, label="线性插值曲线", linestyle="dashed")
plt.legend()
plt.title("Linear Interpolation")
plt.show()



'''
拉格朗日插值
'''
from scipy.interpolate import lagrange
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

'''
牛顿插值
'''
from scipy.interpolate import BarycentricInterpolator
# 创建插值对象
newton_interp = BarycentricInterpolator(x, y)

# 计算新数据点
y_new_newton = newton_interp(x_new)

# 绘图
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new_newton, label="Newton 插值 (Scipy)", linestyle="dashed")
plt.legend()
plt.show()

'''

MES计算均方误差
'''
from sklearn.metrics import mean_squared_error
y_true = y

y_pred_cs = cs(x)
mes_cs = mean_squared_error(y_true=y_true,y_pred=y_pred_cs)
print(f"三次样条插值的均方误差MSE = {mes_cs}")






'''
最小二乘法直线拟合

np.polyfit(x, y, 1) 返回 斜率 𝑎 和 截距 𝑏
1 代表拟合的是 一次多项式（直线）。
'''
import numpy as np
import matplotlib.pyplot as plt

# 给定数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([2.1, 2.9, 3.7, 4.1, 5.3, 5.8])

# 使用 np.polyfit 进行线性最小二乘拟合
a, b = np.polyfit(x, y, 1)  # 1 表示拟合 1 次多项式，即 y = ax + b

# 生成拟合曲线
x_new = np.linspace(0, 5, 100)
y_new = a * x_new + b

# 绘制数据点和拟合直线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_new, label=f"拟合直线: y = {a:.2f}x + {b:.2f}", linestyle="dashed")
plt.legend()
plt.title("最小二乘法拟合直线")
plt.show()

print(f"拟合直线方程: y = {a:.2f}x + {b:.2f}")

'''
多项式最小二乘拟合
'''
# 使用 np.polyfit 进行 2 次多项式（抛物线）拟合
coeffs = np.polyfit(x, y, 2)  # 2 次多项式

# 生成拟合曲线
y_poly = np.polyval(coeffs, x_new)

# 绘制拟合曲线
plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_poly, label=f"二次拟合曲线", linestyle="dashed")
plt.legend()
plt.title("最小二乘法二次拟合")
plt.show()

print(f"拟合二次方程: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}")
'''
lsqcurvefit 拟合
Python 中的 scipy.optimize.curve_fit 可以实现 非线性最小二乘拟合

model_func(x, a, b, c) 是我们要拟合的模型 (函数是什么)y = a * e^{b * x} + c

curve_fit(model_func, x_data, y_data, p0=[1, 1, 1])
p0 是初始参数估计值。
popt 返回最佳拟合参数。
pcov 是协方差矩阵（可用于估计参数的标准误差）。
plt.scatter() 画出数据点，plt.plot() 画出拟合曲线。
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 真实函数
def model_func(x, a, b, c):
    return a * np.exp(b * x) + c

# 生成带噪声的数据
x_data = np.linspace(0, 4, 50)
y_data = model_func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(model_func, x_data, y_data, p0=[1, 1, 1])  # 初始值 p0=[a, b, c]

# 拟合参数
a_fit, b_fit, c_fit = popt
print(f"拟合参数: a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}")

# 绘制数据和拟合曲线
plt.scatter(x_data, y_data, label="带噪声的数据", color="red")
plt.plot(x_data, model_func(x_data, *popt), label="拟合曲线", color="blue")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("非线性最小二乘拟合")
plt.show()
'''
二元函数拟合

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 目标拟合函数
def func(X, a, b, c, d, e, f):
    x, y = X  # X 是 (x, y) 的元组
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

# 生成模拟数据
np.random.seed(0)
x_data = np.linspace(-5, 5, 20)
y_data = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_data, y_data)  # 创建网格
Z = func((X, Y), 1, 2, -1, 3, 4, 5) + np.random.normal(0, 2, X.shape)  # 加入噪声

# 将数据展开成 1D 数组以供 curve_fit 使用
x_flat, y_flat, z_flat = X.ravel(), Y.ravel(), Z.ravel()

# 使用 curve_fit 进行拟合
popt, _ = curve_fit(func, (x_flat, y_flat), z_flat,p0=[])#注意初始值会影响效果，要取好一点，不加初始值也可以

# 获取拟合参数
a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = popt
print(f"拟合参数: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f}, d={d_fit:.4f}, e={e_fit:.4f}, f={f_fit:.4f}")

# 计算拟合结果
Z_fit = func((X, Y), *popt)

# 绘制原始数据
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_flat, y_flat, z_flat, label="原始数据", color="red")
ax.set_title("原始数据")

# 绘制拟合曲面
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_fit, cmap="viridis", alpha=0.7)
ax2.scatter(x_flat, y_flat, z_flat, color="red", label="原始数据")
ax2.set_title("拟合曲面")
plt.show()
