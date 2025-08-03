import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a3 = 8
a2 = 5
a1 = 2
a0 = -1

x = np.linspace(-6, 6, 100)

##正态分布
data = np.random.normal(0, 1, 100)  # 生成 1000 个符合 N(0,1) 的数据


def f_x(x):
    return a3 * (x ** 3) + a2 * (x ** 2) + a1 * x + a0 + data

y = f_x(x)

coeffs = np.polyfit(x, y, 3)  # 3 次多项式

x_new = x

y_poly = np.polyval(coeffs, x_new)

plt.scatter(x, y, color='red', label="原始数据点")
plt.plot(x_new, y_poly, label=f"二次拟合曲线", linestyle="dashed")
plt.legend()
plt.title("最小二乘法二次拟合")
plt.show()

print(f"拟合二次方程: y = {coeffs[0]:.2f}x^3 + {coeffs[1]:.2f}x^2 + {coeffs[2]:.2f}x + {coeffs[3]:.2f}")