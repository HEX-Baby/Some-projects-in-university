from pulp import LpProblem, LpVariable, LpMaximize, lpSum
import numpy as np

problem = LpProblem('2.7',LpMaximize)

Var_num=2

x=[LpVariable(f'x{i+1}',lowBound=0,cat="Integer") for i in range(2)]

c=np.array([2,1])

problem += lpSum(c[i] * x[i] for i in range(Var_num))

A_ub=np.array(
    [
        [0,5],
        [6,2],
        [1,1]
    ]
)

b_ub=np.array(
    [15,24,5]
)


for i in range(len(A_ub)):
    problem += lpSum(A_ub[i][j] * x[j] for j in range(Var_num)) <= b_ub[i]

problem.solve()
print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problem.objective.value())

