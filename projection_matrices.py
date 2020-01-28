import numpy as np
from scipy.sparse import csr_matrix

def projectionMatrix1(n1, n2):
    A = np.zeros((n1, n1 * n2), dtype=int)
    for i in range(n1):
        A[i, n2 * i: n2 * (i + 1)] = 1
    return A


def projectionMatrix2(n1, n2):
    A = np.zeros((n2, n1 * n2), dtype=int)
    for i in range(n2):
        A[i, i: None: n2] = 1 #TODO
    return A


def projectionMatrix3(n1, n2):
    nrows = n1 + n2 - 1
    A = np.zeros((nrows, n1 * n2), dtype=int)
    for j in range(n1 * n2):
        i = j % n2 + j // n2
        A[i, j] = 1
    return A


def projectionMatrix4(n1, n2):
    nrows = n1 + n2 - 1
    A = np.zeros((nrows, n1 * n2), dtype=int)
    for j in range(n1 * n2):
        i = j % n2 - j // n2 + n1 - 1
        A[i, j] = 1
    return A


def combineProjections(n1, n2, m):
    A = np.empty((0, n1 * n2), dtype=int)
    if m[0] == 1:
        A = np.concatenate((A, projectionMatrix1(n1, n2)))
    if m[1] == 1:
        A = np.concatenate((A, projectionMatrix2(n1, n2)))
    if m[2] == 1:
        A = np.concatenate((A, projectionMatrix3(n1, n2)))
    if m[3] == 1:
        A = np.concatenate((A, projectionMatrix4(n1, n2)))
    return csr_matrix(A)