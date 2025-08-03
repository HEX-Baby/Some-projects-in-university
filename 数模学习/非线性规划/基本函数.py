""""
1. SciPy

常用方法：
trust-constr: 适合约束优化问题。
SLSQP（Sequential Least Squares Programming）：解决约束优化问题。
COBYLA（Constrained Optimization By Linear Approximations）：解决非线性约束优化问题。

在使用 scipy.optimize.minimize 进行约束优化时，{'type': 'ineq', 'fun': constraint} 的 不等式约束 是默认表示为 g(x) >= 0 的形式。
COBYLA只支持 不等式约束，即约束必须写成 g(x) >= 0 的形式。

方法	            是否支持约束	    是否需要梯度	    优势	                        劣势
trust-constr	等式 & 不等式	    可选 (推荐提供)	精确度高，适合复杂问题	        计算量大，依赖二阶信息
SLSQP	        等式 & 不等式	    可选 (推荐提供)	处理约束灵活，收敛快	        对初值敏感，高维问题性能较差
COBYLA	        不等式	        不需要	        简单、快速、对不可导函数有效	    仅支持不等式，精度较低
#############################################################################################
1. 使用 trust-constr 方法的代码

from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return x[0]**2 + x[1]**2  # 优化的目标是最小化 x^2 + y^2

# 定义约束
def constraint(x):
    return x[0] + x[1] - 1  # 等式约束: x + y = 1

# 初始值
x0 = [0.5, 0.5]

# 将约束写成一个字典
cons = {'type': 'eq', 'fun': constraint}

# 使用 trust-constr 求解
result = minimize(objective, x0, constraints=cons, method='trust-constr')
print(result)



优化结果 result 包括：
    最优值 result.fun。
    最优解 result.x。
    优化是否成功 result.success
################################################################################################
2. 使用 SLSQP 方法的代码

from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return x[0]**2 + x[1]**2  # 优化的目标是最小化 x^2 + y^2

# 定义约束
def constraint(x):
    return x[0] + x[1] - 1  # 等式约束: x + y = 1

# 初始值
x0 = [0.5, 0.5]

# 将约束写成一个字典
cons = {'type': 'eq', 'fun': constraint}

# 使用 SLSQP 求解
result = minimize(objective, x0, constraints=cons, method='SLSQP')
print(result)



##################################################################################################
3. 使用 COBYLA 方法的代码

from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return x[0]**2 + x[1]**2  # 优化的目标是最小化 x^2 + y^2

# 定义约束
def constraint(x):
    return x[0] + x[1] - 1  # 不等式约束: x + y >= 1

# 初始值
x0 = [0.5, 0.5]

# 将约束写成一个字典
cons = {'type': 'ineq', 'fun': constraint}  # 注意约束类型是 'ineq'

# 使用 COBYLA 求解
result = minimize(objective, x0, constraints=cons, method='COBYLA')
print(result)

"""


##传入额外参数
"""
def objective(x, *args):
    # x 是优化变量（必传）
    # args 是你通过 minimize 的 args 参数传入的额外数据
    ...


例如
from scipy.optimize import minimize
import numpy as np

# 定义目标函数
def objective(x, A, b):
    return np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)  # f(x) = x^T A x + b^T x

# 定义约束
def constraint(x):
    return np.sum(x) - 1  # 约束: x1 + x2 + ... + xn = 1

###如果有多个约束条件##############################################
# 定义约束函数
# 约束1: x1^2 + x2^2 <= 1
def constraint1(x):
    return 1 - (x[0]**2 + x[1]**2)  # g(x) >= 0 -> 1 - (x1^2 + x2^2) >= 0

# 约束2: x1 + x2 = 1
def constraint2(x):
    return x[0] + x[1] - 1  # h(x) = 0

# 初始值
x0 = [0.5, 0.5]

# 定义约束列表
constraints = [
    {'type': 'ineq', 'fun': constraint1},  # 不等式约束1
    {'type': 'eq', 'fun': constraint2}    # 等式约束2
]
###############################################################


# 矩阵 A 和向量 b（额外参数）
A = np.array([[2, 0], [0, 1]])  # 2x2 矩阵
b = np.array([-1, -1])          # 向量 b

# 初始值
x0 = np.array([0.5, 0.5])

# 定义约束
cons = {'type': 'eq', 'fun': constraint}

# 使用 minimize，并通过 args 传入矩阵 A 和向量 b
result = minimize(objective, x0, args=(A, b), constraints=cons, method='SLSQP')

# 输出结果
print("优化结果:", result)

"""

"""
非线性整数规划
使用 Gurobi 的示例:
import gurobipy as grb
import numpy as np

# 设置矩阵维度
n, m = 3, 4  # 例如 3 行 4 列的矩阵

# 创建模型
model = grb.Model("matrix_optimization")

# 使用字典来存储每个变量 x_ij
X = {}
for i in range(n):
    for j in range(m):
        X[i, j] = model.addVar(vtype=grb.GRB.INTEGER, name=f"x_{i}_{j}", lb=0, ub=5)

# 
# # 创建 numpy ndarray 来存储变量
# X = np.empty((n, m), dtype=object)  # 创建一个 n x m 的矩阵
# 
# # 初始化变量，使用 ndarray 存储 Gurobi 变量
# for i in range(n):
#     for j in range(m):
#         X[i, j] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"x_{i}_{j}", lb=0, ub=5)


# 设置目标函数 (最小化 x_ij^2 的总和)
model.setObjective(grb.quicksum(X[i, j]**2 for i in range(n) for j in range(m)), grb.GRB.MINIMIZE)

# 添加约束: 每一行的和应该等于 10
for i in range(n):
    model.addConstr(grb.quicksum(X[i, j] for j in range(m)) == 10, f"row_sum_{i}")

# 求解模型
model.optimize()

# 输出结果
if model.status == grb.GRB.OPTIMAL:
    for i in range(n):
        for j in range(m):
            print(f"x_{i}_{j} = {X[i, j].X}")
    print(f"目标函数值: {model.objVal}")
else:
    print("无法找到最优解")



"""

"""
使用 Gurobi 的示例:
import gurobipy as grb

# 创建模型
m = grb.Model()

# 添加变量 (整数变量)
x1 = m.addVar(vtype=grb.GRB.INTEGER, name="x1")
x2 = m.addVar(vtype=grb.GRB.INTEGER, name="x2")

# 设置目标函数 (最小化 x1^2 + x2^2)
m.setObjective(x1**2 + x2**2, grb.GRB.MINIMIZE)

# 添加约束 (x1 + x2 = 10)
m.addConstr(x1 + x2 == 10)

# 求解
m.optimize()

# 输出结果
print(f"x1: {x1.X}, x2: {x2.X}")


"""