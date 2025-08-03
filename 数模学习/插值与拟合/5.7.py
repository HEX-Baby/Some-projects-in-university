import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 目标拟合函数
def func(X, a, b):
    x, y = X  # X 是 (x, y) 的元组
    return a * x * y / (1 + b * np.sin(x))

# 生成模拟数据
np.random.seed(0)
x_data = np.linspace(-6, 6, 30)
y_data = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x_data, y_data)  # 创建网格
Z = func((X, Y), 2, 3)

# 将数据展开成 1D 数组以供 curve_fit 使用
x_flat, y_flat, z_flat = X.ravel(), Y.ravel(), Z.ravel()

# 使用 curve_fit 进行拟合
popt, _ = curve_fit(func, (x_flat, y_flat), z_flat)

# 获取拟合参数
a_fit, b_fit= popt
print(f"拟合参数: a={a_fit:.4f}, b={b_fit:.4f}")

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
