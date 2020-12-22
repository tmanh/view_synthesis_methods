import numpy as np
from numba import jit


@jit('int32(float32)')
def round(x):
    if x - np.int32(x) > 0.5:
        return np.int32(x) + 1
    return np.int32(x)


@jit('void(int32[:],int32[:],int32[:,:],int32[:,:])')
def meshgrid(x_range, y_range, X, Y):
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            X[j, i] = x_range[i]
            Y[j, i] = y_range[j]


@jit('void(float64[:],float64[:],int32[:,:],int32[:,:])')
def meshgrid_float(x_range, y_range, X, Y):
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            X[j, i] = np.int32(x_range[i])
            Y[j, i] = np.int32(y_range[j])
