from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import numpy as np
problems = LpProblem('2.1',LpMinimize)

Var_num = 6

x = [LpVariable(f"x{i+1}", cat="Binary") for i in range(Var_num)]

problems += lpSum(x[i] for i in range(Var_num))

A_ub=np.array([
    [1,1,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1],
    [1,1,1,0,0,0],
    [0,0,0,0,1,1],
    [1,0,0,0,0,0],
    [0,1,0,1,0,1]
])

b_ub=np.array(
    [1,
     1,
     1,
     1,
     1,
     1,
     1,
     1]
)

for i in range(len(A_ub)):
    problems += lpSum(A_ub[i][j] * x[j] for j in range(Var_num)) >= b_ub[i]

problems.solve()

print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problems.objective.value())


