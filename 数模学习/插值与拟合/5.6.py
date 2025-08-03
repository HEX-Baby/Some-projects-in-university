import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 真实函数
def model_func(x, a, b):
    return 10 * a / (10 * b + (a - 10* b) * np.exp(-a * np.sin(x)))

# 生成带噪声的数据
x_data = np.linspace(1, 20, 20)
y_data = model_func(x_data, 1.1, 0.01) + 0.2 * np.random.normal(size=len(x_data))

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(model_func, x_data, y_data, p0=[1, 1])  # 初始值 p0=[a, b]

# 拟合参数
a_fit, b_fit = popt
print(f"拟合参数: a = {a_fit:.4f}, b = {b_fit:.4f}")

# 绘制数据和拟合曲线
plt.scatter(x_data, y_data, label="带噪声的数据", color="red")
plt.plot(x_data, model_func(x_data, *popt), label="拟合曲线", color="blue")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("非线性最小二乘拟合")
plt.show()
