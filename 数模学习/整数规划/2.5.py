from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger
import numpy as np
#####################(1)#########################
# A_ub=np.array(
#     [
#         [1,2,0,0,0,0,1],
#         [0,0,3,2,1,0,1],
#         [4,1,0,2,4,6,1]
#     ]
# )
# Var_num=7
# c=np.array([1]*7)
# problem=LpProblem('2.5',LpMinimize)
# x=[LpVariable(f'x{i+1}',lowBound=0,cat="Integer") for i in range(Var_num)]
# b_ub=np.array([100]*3)
#
# problem += lpSum(c[i]*x[i] for i in range(Var_num))
#
# for i in range(len(A_ub)):
#     problem += lpSum(A_ub[i][j] * x[j] for j in range(Var_num)) >= b_ub[i]
#
# problem.solve()
#
#
# print("Optimal Solution:")
# for v in x:
#     print(f"{v.name} = {v.varValue}")
# print("Optimal Value:", problem.objective.value())


################(2)####################
A_ub=np.array(
    [
        [1,2,0,0,0,0,1],
        [0,0,3,2,1,0,1],
        [4,1,0,2,4,6,1]
    ]
)
Var_num=7
c=np.array([1]*7)

problem=LpProblem('2.5',LpMinimize)

x=[LpVariable(f'x{i+1}',lowBound=0,cat="Integer") for i in range(Var_num)]

y=[LpVariable(f'y{i+1}',cat="Binary") for i in range(Var_num)]

M=1e6

b_ub=np.array([100]*3)

problem += lpSum(c[i]*x[i] for i in range(Var_num))

for i in range(len(A_ub)):
    problem += lpSum(A_ub[i][j] * x[j] for j in range(Var_num)) >= b_ub[i]

for i in range(Var_num):
    problem += x[i] <= M*y[i]

problem += lpSum(y[i] for i in range(Var_num)) == 3

problem.solve()


print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problem.objective.value())

for v in y:
    print(f"{v.name} = {v.varValue}")
