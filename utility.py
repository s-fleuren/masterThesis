import numpy as np
from scipy import sparse


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p. Taken from
    https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def unpackbits16(n, l):
    n1 = np.array([n//256], dtype=np.uint8)
    n2 = np.array([n], dtype=np.uint8)
    b = np.concatenate((np.unpackbits(n1), np.unpackbits(n2)))
    return b[16 - l:]


def problem_reduction_single(A, y, i, val):
    new_A = sparse.hstack([A[:, :i], A[:, i + 1:]])
    y_update = - val * A.getcol(i).A.reshape(y.size)
    new_y = y + y_update
    return new_A, new_y


def split_nu(nu):
    N = nu.size
    alpha = np.zeros(N)
    beta = np.zeros(N)
    for i in range(N):
        if nu[i] > 0:
            beta[i] = nu[i]
        elif nu[i] < 0:
            alpha[i] = -nu[i]
    return alpha, beta


def rounding(x, A, y):
    M, N = A.shape
    new_x = x.copy()
    distance_to_boundary = np.zeros_like(x)
    for i in range(N):
        distance_to_boundary[i] = min(x[i], 1 - x[i])
    order = distance_to_boundary.argsort()
    for j in range(N):
        i = order[j]
        new_x[i] = 0
        cost0 = np.linalg.norm(A @ new_x - y)
        new_x[i] = 1
        cost1 = np.linalg.norm(A @ new_x - y)
        if cost0 < cost1:
            new_x[i] = 0
    return new_x


def compare_vectors(x1, x2):
    counter = 0
    for i in range(x1.size):
        if x1[i] == x2[i]:
            counter += 1
    return counter