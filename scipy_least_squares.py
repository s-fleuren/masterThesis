from scipy.optimize import least_squares
# from projection_matrices import *


def fun(x, A, y):
    return A @ x - y


def jac_A(x, A, y):
    return A


def compute_constrained_least_squares_solution(A, y, x_0, tol, verbose=0):
    return least_squares(fun, x_0, jac=jac_A, bounds=(0, 1), method='trf', verbose=verbose, jac_sparsity='sparse matrix',
                         gtol=tol, ftol=None, xtol=None, args=(A, y))