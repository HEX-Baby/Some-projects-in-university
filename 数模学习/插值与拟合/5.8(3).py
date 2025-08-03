import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.optimize import curve_fit
x = np.linspace(-2,4.9,24)

y = np.array([0.1029, 0.1174, 0.1316, 0.1448, 0.1556, 0.1662, 0.1733, 0.1775,
              0.1785, 0.1764, 0.1711, 0.1630, 0.1526, 0.1402, 0.1266, 0.1122,
              0.0977, 0.0835, 0.0702, 0.0588, 0.0479, 0.0373, 0.0291, 0.0224])
def f(x,sigema,mu):
    return np.exp( - ((x - mu) ** 2) / (2 * sigema ** 2)) / (np.sqrt(2 * np.pi) * sigema)

popt, pcov = curve_fit(f, x, y)

sigema, mu= popt

y_pred = f(x,*popt)

MSE = mean_squared_error(y_true=y,y_pred=y_pred)

print(f"拟合参数: a = {sigema:.4f}, b = {mu:.4f}")
plt.scatter(x, y, label="带噪声的数据", color="red")
plt.plot(x, f(x, *popt), label="拟合曲线", color="blue")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("非线性最小二乘拟合")
plt.show()

print(f"MSE = {MSE}")