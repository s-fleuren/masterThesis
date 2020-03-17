from scipy.optimize import least_squares




def fun(x, A, y):
    return A @ x - y


def jac_A(x, A, y):
    return A


def compute_constrained_least_squares_solution(A, y, x_0, tol, verbose=0):
    """
    Wrapper around the Scipy least squares solver to find an epsilon-pseudo-optimal solution of CCLS.
    :param A: The projection matrix of the discrete tomography problem. Should be given in a sparse matrix format.
    :param y: The projection data of the discrete tomography problem.
    :param x_0: Initial guess for an optimal solution of the relaxed discrete tomography problem.
    :param tol: Tolerance used for termination. To find an epsilon-pseudo-optimal solution, set tol=epsilon. See the
                gtol option in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html,
                after "For 'trf'".
    :param verbose: Level of algorithm’s verbosity. See
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html.
                    Default is 0 (silent)
    :return: scipy.optimize.OptimizeResult
    """
    return least_squares(fun, x_0, jac=jac_A, bounds=(0, 1), method='trf', verbose=verbose, jac_sparsity='sparse matrix',
                         gtol=tol, ftol=None, xtol=None, args=(A, y))