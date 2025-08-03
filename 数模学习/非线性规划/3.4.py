# from scipy.optimize import minimize
# import numpy as np
#
# # 初始值
# x0 = np.random.uniform(low=0, high=10, size=100)
# if np.sum((100 - np.arange(100)) * x0) > 1000:
#     x0 = x0 * (1000 / np.sum((100 - np.arange(100)) * x0))  # 归一化
#
# # 目标函数
# def objective(x):
#     return -np.sum(np.sqrt(x))  # 使用 np.sqrt 代替 x**0.5，更清晰
#
# # 定义约束条件
# constraints = []
#
# # 线性递增约束条件
# for i in range(1, 5):
#     def constraint_linear(x, i=i):  # 必须使用缺省参数避免闭包问题
#         return i * 10 - np.sum(np.arange(1, i + 1) * x[:i])
#     constraints.append({'type': 'ineq', 'fun': constraint_linear})
#
# # 非线性约束条件
# def constraint_sum(x):
#     return 1000 - np.sum((100 - np.arange(100)) * x)
# constraints.append({'type': 'ineq', 'fun': constraint_sum})
#
# # 非负约束条件
# bounds = [(0, None)] * 100  # 直接通过 bounds 实现非负约束
#
# # 求解
# result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
#
# # 输出结果
# if result.success:
#     print("最优解:", result.x)
#     print("最优目标函数值:", -result.fun)
# else:
#     print("优化失败:", result.message)
import gurobipy as grb
import numpy as np

n = 100

model = grb.Model("3.4")

# 创建变量
x = np.empty(n, dtype=object)
y = np.empty(n, dtype=object)  # 辅助变量 y[i]

for i in range(n):
    x[i] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"x_{i+1}", lb=0)
    y[i] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"y_{i+1}", lb=0)

# 添加二次约束：y[i]^2 = x[i]
for i in range(n):
    model.addConstr(x[i] == y[i] * y[i], name=f"quad_constr_{i+1}")

# 设置目标函数：最大化 sum(y[i])
model.setObjective(grb.quicksum(y[i] for i in range(n)), grb.GRB.MAXIMIZE)

# 添加线性递增约束
for i in range(1, 5):
    model.addConstr(grb.quicksum(j * x[j-1] for j in range(1, i+1)) <= 10 * i)

# 添加非线性约束（手动线性化）
model.addConstr(grb.quicksum((100 - i) * x[i] for i in range(100)) <= 1000)
model.setParam('TimeLimit', 300)  # 设置时间限制
# 求解
model.optimize()

# 输出结果
if model.status == grb.GRB.OPTIMAL:
    for i in range(n):
        print(f"x_{i} = {x[i].X}, sqrt(x_{i}) = {y[i].X}")
    print(f"目标函数值: {model.objVal}")
else:
    print("无法找到最优解")


