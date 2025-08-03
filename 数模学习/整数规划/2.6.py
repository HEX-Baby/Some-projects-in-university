from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import numpy as np

problem=LpProblem('2.6',LpMinimize)

Var_num=5

x=[LpVariable(f'x{i+1}',lowBound=0,cat="Integer") for i in range(Var_num)]

c=np.array([20,90,80,70,30])

A_ub1=np.array(
    [
        [1,1,0,0,1],
        [0,0,1,1,0]
    ]
)
A_ub2=np.array(
    [
        [3,0,2,0,0],
        [0,3,0,2,1]
    ]
)

b_ub1=np.array([30,30])

b_ub2=np.array([120,48])

problem += lpSum(c[i] * x[i] for i in range(Var_num))

for i in range(len(A_ub1)):
    problem += lpSum(A_ub1[i][j] * x[j] for j in range(Var_num)) >= b_ub1[i]

for i in range(len(A_ub2)):
    problem += lpSum(A_ub2[i][j] * x[j] for j in range(Var_num)) <= b_ub2[i]

problem.solve()

print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problem.objective.value())