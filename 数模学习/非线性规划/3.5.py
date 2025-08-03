import gurobipy as grb

# 创建模型
model = grb.Model("Quadratic_Optimization")


# 定义变量
x1 = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="x1")  # x1 >= 0
x2 = model.addVar(vtype=grb.GRB.CONTINUOUS, name="x2")        # 默认 lb = -∞
x3 = model.addVar(vtype=grb.GRB.CONTINUOUS, name="x3")        # 默认 lb = -∞
x = [x1, x2, x3]

# 设置目标函数: 2*x1 + 3*x1^2 + 3*x2 + x2^2 + x3
obj = grb.QuadExpr()
obj += 2 * x1 + 3 * x1**2
obj += 3 * x2 + x2**2
obj += x3
model.setObjective(obj, grb.GRB.MAXIMIZE)

# 添加约束条件
# 约束 1: x1^2 + x3 == 2
model.addConstr(x1**2 + x3 == 2, name="Constraint_1")

# 约束 2: A_ub1
A_ub1 = [
    [1, 1, 1, 2, 2, 0],
    [1, 1, -1, 1, 1, 0],
    [2, 2, 1, 1, 0, 0]
]
b_ub1 = [10, 50, 40]

for i in range(len(A_ub1)):
    expr = grb.LinExpr()
    for j in range(len(x)):  # 遍历 x1, x2, x3
        expr += A_ub1[i][j] * x[j]  # 线性部分
        expr += A_ub1[i][j + 3] * x[j]**2  # 二次部分
    model.addConstr(expr <= b_ub1[i], name=f"Constraint_2_{i+1}")

# 约束 3: A_ub2
A_ub2 = [1, 2, 0, 0, 0, 0]
b_ub2 = [1]

expr = grb.LinExpr()
for i in range(len(x)):
    expr += A_ub2[i] * x[i]  # 线性部分
    expr += A_ub2[i + 3] * x[i]**2  # 二次部分
model.addConstr(expr >= b_ub2[0], name="Constraint_3")

# 求解模型
model.optimize()

# 输出结果
if model.status == grb.GRB.OPTIMAL:
    for i in range(len(x)):
        print(f"x_{i+1} = {x[i].X}")
    print(f"目标函数值: {model.objVal}")
else:
    print("无法找到最优解")
#############################################################################

import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective(vars):
    x1, x2, x3 = vars
    return -(2 * x1 + 3 * x1**2 + 3 * x2 + x2**2 + x3)  # 取负号，转化为最小化问题

# 定义约束条件
def constraint1(vars):
    x1, x2, x3 = vars
    return x1**2 + x3 - 2  # 等式约束，f(x) == 0

def constraint2_1(vars):
    x1, x2, x3 = vars
    return 10 - (1 * x1 + 1 * x2 + 1 * x3 + 2 * x1**2 + 2 * x2**2 + 0 * x3**2)  # 不等式约束，f(x) >= 0

def constraint2_2(vars):
    x1, x2, x3 = vars
    return 50 - (1 * x1 + 1 * x2 - 1 * x3 + 1 * x1**2 + 1 * x2**2 + 0 * x3**2)

def constraint2_3(vars):
    x1, x2, x3 = vars
    return 40 - (2 * x1 + 2 * x2 + 1 * x3 + 1 * x1**2 + 0 * x2**2 + 0 * x3**2)

def constraint3(vars):
    x1, x2, x3 = vars
    return (1 * x1 + 2 * x2 + 0 * x3 + 0 * x1**2 + 0 * x2**2 + 0 * x3**2) - 1

# 设置初始猜测值
x0 = [0.5, 0.5, 0.5]

# 定义变量的边界
bounds = [(0, None),  # x1 >= 0
          (None, None),  # x2 无边界
          (None, None)]  # x3 无边界

# 定义约束条件列表
constraints = [
    {'type': 'eq', 'fun': constraint1},  # 等式约束
    {'type': 'ineq', 'fun': constraint2_1},  # 不等式约束
    {'type': 'ineq', 'fun': constraint2_2},
    {'type': 'ineq', 'fun': constraint2_3},
    {'type': 'ineq', 'fun': constraint3}
]

# 求解优化问题
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# 输出结果
if result.success:
    x1_opt, x2_opt, x3_opt = result.x
    print(f"最优解：x1 = {x1_opt:.4f}, x2 = {x2_opt:.4f}, x3 = {x3_opt:.4f}")
    print(f"目标函数值：{-result.fun:.4f}")  # 转回最大化问题的结果
else:
    print("优化问题无法找到可行解")
