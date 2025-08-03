from pulp import LpProblem, LpVariable, LpMaximize, lpSum
import numpy as np

c=np.array(
    [4,2,3,4,
     6,4,5,5,
     7,6,7,6,
     7,8,8,6,
     7,9,8,6,
     7,10,8,6]
)
problems = LpProblem('2.3',LpMaximize)
Var_num = 6*4

x=[LpVariable(f'x{i+1}', cat='Binary') for i in range(Var_num)]

problems += lpSum(c[i] * x[i] for i in range(Var_num))

A_eq=np.array(
    [
        [1]*4+[0]*(Var_num - 4),
        [0]*4+[1]*4+[0]*(Var_num - 8),
        [0] * 8 + [1] * 4 + [0] * (Var_num - 12),
        [0] * 12 + [1] * 4 + [0] * (Var_num - 16),
        [0] * 16 + [1] * 4 + [0] * (Var_num - 20),
        [0] * 20 + [1] * 4 + [0] * (Var_num - 24),
    ]
)

b_eq=np.array([1]*6)

#等式条件

for i in range(len(A_eq)):
    problems += lpSum(A_eq[i][j] * x[j] for j in range(Var_num)) == b_eq[i]

A_ub=np.array(
    [
        [1,0,0,0]*6,
        [0,1,0,0]*6,
        [0,0,1,0]*6,
        [0,0,0,1]*6
    ]
)

b_ub=np.array([1]*4)

for i in range(len(A_ub)):
    problems += lpSum(A_ub[i][j] * x[j] for j in range(Var_num)) >= b_ub[i]

problems.solve()

print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problems.objective.value())