# from scipy.optimize import linprog
# #scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs', options=None)
# c=[1-300/6000,
#    1-7*321*1e4,
#    -6*250/4000,
#    -4*783/7000,
#    -7*200/4000,
#    -10*300/6000
#    ,-9*321/1e4,
#    2-0.35-8*250/4000,
#    2.8-0.5-12*321/1e4-11*783/7000]
#
# for i in c:
#     i=i*(-1)
#
# A_ub=[
#     [5,0,0,0,0,10,0,0,0],
#     [0,7,0,0,0,0,9,0,12],
#     [0,0,6,0,0,0,0,8,0],
#     [0,0,0,4,0,0,0,0,11],
#     [0,0,0,0,7,0,0,0,0]
# ]
#
# b_ub=[6000,1e4,4000,7000,4000]
#
# A_eq=[
#     [1,1,-1,-1,-1,0,0,0,0],
#     [0,0,0,0,0,1,1,-1,0]
# ]
#
# b_eq=[0,0]
#
# res=linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,integrality=[1,1,1,1,1,1,1,1,1])
#
# if res.success:
#     print("最优解为：",-res.fun)
#     for i in range(9):
#         print("x[%d]"%(i),res.x[i])
# else:
#     print("无最优解")
#无法处理整数规划
from pulp import LpProblem, LpVariable, LpMaximize, lpSum
import numpy as np

# 定义问题（最大化目标）
problems = LpProblem("1.3", LpMaximize)

# 定义变量
x = [LpVariable(f'x{i}', lowBound=0, cat="Integer") for i in range(9)]

# 定义目标函数系数
c = [
    1 - 5 * 300 / 6000,
    1 - 7 * 321 * 1e4,
    -6 * 250 / 4000,
    -4 * 783 / 7000,
    -7 * 200 / 4000,
    -10 * 300 / 6000,
    -9 * 321 / 1e4,
    2 - 0.35 - 8 * 250 / 4000,
    2.8 - 0.5 - 12 * 321 / 1e4 - 11 * 783 / 7000,
]

# 添加目标函数
problems += lpSum(c[i] * x[i] for i in range(9)), "Objective Function"

# 定义不等式约束
A_ub = np.array([
    [5, 0, 0, 0, 0, 10, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 9, 0, 12],
    [0, 0, 6, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 11],
    [0, 0, 0, 0, 7, 0, 0, 0, 0]
])
b_ub = [6000, 1e4, 4000, 7000, 4000]

# 添加不等式约束
for i in range(len(A_ub)):
    problems += lpSum(A_ub[i][j] * x[j] for j in range(9)) <= b_ub[i], f"Inequality_{i+1}"

# 定义等式约束
A_eq = np.array([
    [1, 1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, -1, 0]
])
b_eq = [0, 0]

# 添加等式约束
for i in range(len(A_eq)):
    problems += lpSum(A_eq[i][j] * x[j] for j in range(9)) == b_eq[i], f"Equality_{i+1}"

# 求解问题
problems.solve()

# 输出结果
print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problems.objective.value())

