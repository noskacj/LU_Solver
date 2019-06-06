import numpy as np
from scipy import linalg
import time


A = np.random.uniform(0,10,1000000).reshape(1000,1000)
b = np.ones((1000, 1))


def forward_solve(L, b):
    l = len(b)
    y = np.zeros(l)
    y[0] = b[0] / L[0, 0]
    for i in range(1, l):
        y[i] = (b[i] - np.dot(L[i,: i],y[: i])) / L[i, i]

    return y


def backward_solve(U, y):
    l = len(y)
    x = np.zeros(l)
    x[l-1] = y[l-1] / U[l-1, l-1]
    for i in range(2, l+1, 1):
        x[-i] = (y[-i] - np.dot(U[-i, -i:],x[-i:])) / U[-i, -i]
    return x


def LU_invert(A, b): # Doesn't just invert, also solves for b
    P, L, U = linalg.lu(A)
    y = forward_solve(L=L, b=b)
    x = backward_solve(U=U, y=y)

    return x


start = time.clock()
for i in range(1000):
    a = LU_invert(A=A, b=b)

print("time for LU (this code): ", time.clock() - start)

start = time.clock()
for i in range(1000):
    a_inv = np.linalg.inv(A)
    solu = a_inv.dot(b)
print("time for inverting (numpy): ", time.clock() - start)

start = time.clock()

for i in range(1000):
    lu, piv = linalg.lu_factor(A)
    x = linalg.lu_solve((lu, piv), b)
print("time for fancy scipy LU solution: ", time.clock() - start)

start = time.clock()
for i in range(1000):
    a_inv = linalg.inv(A)
    solu = a_inv.dot(b)
print("time for inverting (scipy): ", time.clock() - start)
