from projection_matrices import *
from scipy_least_squares import *
from utility import *

def get_all_unique_x(d):
    output = []
    for k in d.keys():
        if len(d[k]) == 0:
            print('error')
        elif len(d[k]) == 1:
            output.append(d[k][0])
    return output


tol = 1e-4
for m in range(2, 5):
    for n in range(2, 5):
        used_projections = [1] * m + [0] * (4 - m)
        A = combineProjections(n, n, used_projections)
        M = A.shape[0]
        N = n ** 2
        x_0 = 0.5 * np.ones(N - 1)
        T = {}
        for i in range(2 ** N):
            x = unpackbits16(i, N)
            y = A @ x
            T.setdefault(tuple(y), [])
            T[tuple(y)].append(i)
        uniques = get_all_unique_x(T)
        for i in uniques:
            x = unpackbits16(i, N)
            y = A @ x
            for j in range(N):
                if x[j] == 0:
                    temp_A, temp_y = problem_reduction_single(A, y, j, 1)
                    result = compute_constrained_least_squares_solution(temp_A, temp_y, x_0, tol)
                    nu = temp_A.transpose() @ (temp_y - temp_A @ result.x)
                    alpha, beta = split_nu(nu)
                    langrangian_value = result.cost - alpha @ result.x + beta @ (result.x - np.ones(result.x.size))
                    if langrangian_value < 0:
                        print("Second solution found:\n" + str(np.insert(result.x, j, 1)))
                        print("Cost: " + str(result.cost))
                        print("Langrangian value: " + str(langrangian_value))
                        print("m, n: " + str(m) + " " + str(n))
                        print("Original solution: " + str(x) + "\n")
                elif x[j] == 1:
                    temp_A, temp_y = problem_reduction_single(A, y, j, 0)
                    result = compute_constrained_least_squares_solution(temp_A, temp_y, x_0, tol)
                    nu = temp_A.transpose() @ (temp_y - temp_A @ result.x)
                    alpha, beta = split_nu(nu)
                    langrangian_value = result.cost - alpha @ result.x + beta @ (result.x - np.ones(result.x.size))
                    if langrangian_value < 0:
                        print("Second solution found:\n" + str(np.insert(result.x, j, 0)))
                        print("Cost: " + str(result.cost))
                        print("Langrangian value: " + str(langrangian_value))
                        print("m, n: " + str(m) + " " + str(n))
                        print("Original solution: " + str(x) + "\n")
        print("done with m=" + str(m) + ", n=" + str(n))

