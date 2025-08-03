from scipy.optimize import minimize
import numpy as np
def objective(x,A):
    return np.dot(x.T,np.dot(A,x))

def constraits(x):
    return np.sum(x**2)-1

x=np.array([1,0,0])

A=np.array(
    [
        [1,4,5],
        [4,2,6],
        [5,6,3]
    ]
)

cons={'type':'eq','fun':constraits}

result=minimize(objective,x,args=(A),constraints=cons,method='SLSQP')
print(result)