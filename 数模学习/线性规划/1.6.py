# from pulp import LpProblem, LpVariable, LpMaximize, lpSum
# import numpy as np
# problems =LpProblem('1.6',LpMaximize)
# Var_num=6
# x=[LpVariable(f'x{i+1}',lowBound=0) for i in range(Var_num)]
#
#
#
# c=np.array(
#     [
#         4, 12, 16,
#         -5, 3, 7
#     ]
# )
# problems += lpSum(c[i] * x[i] for i in range(Var_num))
# A_ub1=np.array(
#     [
#         [1,0,0]*2,
#         [0,1,0]*2,
#         [0,0,1]*2,
#     ]
# )
#
# b_ub1=np.array([
#     500,
#     750,
#     625
# ])
#
# A_ub2=np.array(
#     [
#         [1,1,1,0,0,0],
#         [0,0,0,1,1,1],
#         [0.5,-0.5,-0.5,0,0,0],
#         [-0.25,0.75,-0.25,0,0,0],
#         [0.1,0.1,-0.9,0,0,0],
#         [0,0,0,-0.6,0.4,0.4],
#         [0,0,0,0.4,-0.6,0.4],
#         [0,0,0,-0.15,-0.15,0.85]
#     ]
# )
#
# b_ub2=np.array([
#     600,
#     800,
#     0,
#     0,
#     0,
#     0,
#     0,
#     0
#     ])
#
# for i in range(len(A_ub1)):
#     problems += lpSum(A_ub1[i][j] * x[j] for j in range(Var_num)) <= b_ub1[i]
#
# for i in range(len(A_ub2)):
#     problems += lpSum(A_ub2[i][j] * x[j] for j in range(Var_num)) >= b_ub2[i]
#
# problems.solve()
#
# print("Optimal Solution:")
# for v in x:
#     print(f"{v.name} = {v.varValue}")
# print("Optimal Value:", problems.objective.value())

from scipy.optimize import linprog
import numpy as np
c=np.array(
    [
        4, 12, 16,
        -5, 3, 7
    ]
)
c=c*(-1)
A_ub1=np.array(
    [
        [1,0,0]*2,
        [0,1,0]*2,
        [0,0,1]*2,
    ]
)

b_ub1=np.array([
    500,
    750,
    625
])

A_ub2=np.array(
    [
        [1,1,1,0,0,0],
        [0,0,0,1,1,1],
        [0.5,-0.5,-0.5,0,0,0],
        [-0.25,0.75,-0.25,0,0,0],
        [0.1,0.1,-0.9,0,0,0],
        [0,0,0,-0.6,0.4,0.4],
        [0,0,0,0.4,-0.6,0.4],
        [0,0,0,-0.15,-0.15,0.85]
    ]
)

b_ub2=np.array([
    600,
    800,
    0,
    0,
    0,
    0,
    0,
    0
    ])

A_ub2=-1*A_ub2
A_ub1=np.concatenate((A_ub1,A_ub2),axis=0)
b_ub2=-1*b_ub2
b_ub1=np.concatenate((b_ub1,b_ub2),axis=0)

res=linprog(c=c,A_ub=A_ub1,b_ub=b_ub1)
if res.success:
    for i in range(6):
        print(f'x{i+1}:',res.x[i])

    print('最优解为：',-res.fun)
else:
    print("NO")