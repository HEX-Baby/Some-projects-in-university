#求解 min z = |x1| + 2|x2| + 3|x3| + 4|x4|
#令 u[i]=(|x[i]| + x[i]) / 2  , v[i]=(|x[i]| - x[i]) / 2   , x[i] = u[i] - v[i]  , |x[i]| = u[i] + v [i]


'''
scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs', options=None)
'''

from scipy.optimize import linprog
#   min z = |x1| + 2|x2| + 3|x3| + 4|x4| = u1 + 2u2 + 3u3 + 4u4 + v1 + 2v2 + 3v3 + 4v4
#   x1-x2-x3+x4=u1-u2-u3+u4 - v1 + v2 + v3 - v4
#   x1 - x2 + x3 - 3x4 = u1 - u2 + u3 - 3u4 - v1 + v2 - v3 + 3v4 = 1
#   x1 - x2 - 2x3 + 3x4 = u1 - u2 - 2u3 + 3u4 - v1 + v2 + 2v3 - 3v4 = -1/2
c=[1,2,3,4,1,2,3,4]

A_eq=[
    [1 ,-1 ,-1 ,1,-1,1,1,-1],
    [1,-1,1,-3,-1,1,-1,3],
    [1,-1,-2,3,-1,1,2,-3]
]
b_eq=[0,1,-1/2]

res=linprog(c=c,A_eq=A_eq,b_eq=b_eq)

x=[]

for i in range(4):
    x.append(res.x[i]+res.x[4+i])

for i in range(4):
    print(x[i])

print("最优解为：",res.fun)






