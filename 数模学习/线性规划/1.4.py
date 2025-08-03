from pulp import LpProblem, LpVariable, LpMaximize, lpSum
x=[LpVariable(f"x{i+1}",lowBound=0) for i in range(12)]

problems = LpProblem('1.4',LpMaximize)

c=[
    3100,3100,3100,
    3800,3800,3800,
    3500,3500,3500,
    2850,2850,2850
]

problems += lpSum(c[i] * x[i] for i in range(12))

A_eq=[
    [1/10,-1/16,0]*4,
    [1/10,0,-1/8]*4
]

b_eq=[
    0,
    0
]

A_ub=[
    [1,0,0]*4,
    [0,1,0]*4,
    [0,0,1]*4,
    [480,0,0,650,0,0,580,0,0,390,0,0],
    [0,480,0,0,650,0,0,580,0,0,390,0],
    [0,0,480,0,0,650,0,0,580,0,0,390],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
]

b_ub=[
    10,
    16,
    8,
    6800,
    8700,
    5300,
    18,
    15,
    23,
    12,
]

for i in range(len(A_ub)):
    problems += lpSum(x[j] * A_ub[i][j] for j in range(12)) <= b_ub[i]

for i in range(len(A_eq)):
    problems += lpSum(x[j] * A_eq[i][j] for j in range(12)) == b_eq[i]

problems.solve()

print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problems.objective.value())

# from scipy.optimize import linprog
# c=[
#     -3100,-3100,-3100,
#     -3800,-3800,-3800,
#     -3500,-3500,-3500,
#     -2850,-2850,-2850
# ]
#
# A_eq=[
#     [1/10,-1/16,0]*4,
#     [1/10,0,-1/8]*4
# ]
#
# b_eq=[
#     0,
#     0
# ]
#
# A_ub=[
#     [1,0,0]*4,
#     [0,1,0]*4,
#     [0,0,1]*4,
#     [480,0,0,650,0,0,580,0,0,390,0,0],
#     [0,480,0,0,650,0,0,580,0,0,390,0],
#     [0,0,480,0,0,650,0,0,580,0,0,390],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
# ]
#
# b_ub=[
#     10,
#     16,
#     8,
#     6800,
#     8700,
#     5300,
#     18,
#     15,
#     23,
#     12,
# ]
# res=linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq)
#
# if res.success:
#     for i in range(12):
#         print(res.x[i])
#     print(-res.fun)
#
# else:
#     print("无解")