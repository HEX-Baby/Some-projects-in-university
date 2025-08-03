import gurobipy as grb
import numpy as np

n = 5  # 决策变量个数
model = grb.Model("3.3")

# 定义变量数组
x = np.empty(n, dtype=object)
for i in range(n):
    x[i] = model.addVar(vtype=grb.GRB.INTEGER, name=f"x_{i}", lb=0, ub=99)

# 定义参数矩阵
c = np.array([
    [1, 1, 3, 4, 2],
    [-8, -2, -3, -1, -2]
])

# 设置目标函数
model.setObjective(
    grb.quicksum(c[0, i] * (x[i]**2) + c[1, i] * x[i] for i in range(n)),
    grb.GRB.MAXIMIZE
)

# 定义约束矩阵和约束向量
A = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 1, 6],
    [2, 1, 6, 0, 0],
    [0, 0, 1, 1, 5]
])
b = np.array([400, 800, 200, 200])

# 添加约束
for i in range(4):
    model.addConstr(grb.quicksum(A[i][j] * x[j] for j in range(n)) <= b[i])

# 求解模型
model.optimize()

# 输出结果
if model.status == grb.GRB.OPTIMAL:
    for i in range(n):
        print(f"x_{i} = {x[i].X}")
    print(f"目标函数值: {model.objVal}")
else:
    print("无法找到最优解")

