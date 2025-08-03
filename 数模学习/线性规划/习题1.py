'''
#使用from scipy.optimize import linprog
from scipy.optimize import linprog

# 定义目标函数的系数
c = [-3, 1, 1]

# 定义不等式约束
A_ub = [
    [1, -2, 1],
    [4, -1, -2]
]
b_ub = [11, -3]

# 定义等式约束
A_eq = [[-2, 0, 1]]
b_eq = [1]

# 定义变量的边界 (如果没有特殊要求，可以省略)
bounds = [(None, None), (None, None), (None, None)]

# 调用 linprog
result_1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

# 检查求解是否成功
if result_1.success:
    print("Optimal solution found!")
    print("Optimal values of variables:")
    for i, val in enumerate(result_1.x):
        print(f"x{i + 1} = {val}")
else:
    print("Optimization failed:", result_1.message)
'''

# from pulp import LpProblem, LpVariable, LpMaximize, lpSum #LpMaximize可以换成LpMinimize
# 使用pulp
# problem=LpProblem("习题1",LpMaximize)
#
# x1=LpVariable('x1',lowBound=0)
# x2=LpVariable('x2',lowBound=0)
# x3=LpVariable('x3',lowBound=0)
#
# problem += 3*x1-x2-x3
#
# # 添加约束条件
# problem += x1-2*x2+x3 <= 11
# problem += -4*x1 + x2 + 2*x3 >= 3
# problem += -2*x1 + x3 ==1
#
# problem.solve()
#
# print("Status:", problem.status)
# print("x1 =", x1.varValue)
# print("x2 =", x2.varValue)
# print("Objective Value =", problem.objective.value())