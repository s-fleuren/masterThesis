import numpy as np
from projection_matrices import *
from utility import *


class Iterate:
    def __init__(self, initial_solution):
        self.solution = initial_solution
        self.is_active = np.zeros_like(self.solution, dtype=int)

    def gradientProjectionStep(self, A, y):
        N = self.solution.size
        infty = 2 ** 63
        c = - A.transpose() * y
        gradient = A.transpose() * (A * self.solution) + c
        for i in range(N):
            if self.is_active[i]:
                gradient[i] = 0
        # Compute breakpoints
        breakpoints = np.zeros(N)
        for i in range(N):
            if gradient[i] < 0:  # TODO: more stable bound if possible
                breakpoints[i] = (self.solution[i] - 1) / gradient[i]
            elif gradient[i] > 0:
                breakpoints[i] = self.solution[i] / gradient[i]
            else:
                breakpoints[i] = infty
        order = breakpoints.argsort()

        # Search on the first line-piece
        delta_t = breakpoints[order[0]]
        current_direction = -gradient
        first_derivative_f = c.transpose() @ current_direction + \
                             (self.solution.transpose() @ A.transpose()) @ (A @ current_direction)
        if first_derivative_f >= 0:  # Current solution is minimal
            return
        second_derivative_f = (current_direction.transpose() @ A.transpose()) @ (A @ current_direction)
        delta_t_opt = - first_derivative_f / second_derivative_f
        if delta_t_opt < delta_t:
            self.solution = self.solution + delta_t_opt * current_direction
            return

        # Search on the remaining line pieces
        for j in range(N - 1):
            i = order[j]
            next_i = order[j + 1]
            if breakpoints[next_i] == infty:
                # No significant improvement in remaining direction
                return
            self.solution = self.solution + delta_t * current_direction
            # Constraint of x_i becomes active
            self.solution[i] = np.round(self.solution[i])
            self.is_active[i] = 1
            delta_t = breakpoints[next_i] - breakpoints[i]
            # TODO: if delta_t == 0:
            current_direction[i] = 0
            first_derivative_f = c.transpose() @ current_direction + \
                                 (self.solution.transpose() @ A.transpose()) @ (A @ current_direction)
            if first_derivative_f >= 0:  # Current solution is minimal
                return
            second_derivative_f = (current_direction.transpose() @ A.transpose()) @ (A @ current_direction)
            delta_t_opt = - first_derivative_f / second_derivative_f
            if delta_t_opt < delta_t:
                self.solution = self.solution + delta_t_opt * current_direction
                return


used_projections = [1] * 2 + [0] * 2
A = combineProjections(2, 2, used_projections)
y = np.array([1, 1, 2, 0])
x = Iterate(np.array([0.5, 0.5, 0.5, 0.5]))
x.gradientProjectionStep(A, y)
print(x.solution)
print(x.is_active)
x.gradientProjectionStep(A, y)
print(x.solution)
print(x.is_active)
x.gradientProjectionStep(A, y)
print(x.solution)
print(x.is_active)
x.gradientProjectionStep(A, y)
print(x.solution)
print(x.is_active)
x.gradientProjectionStep(A, y)
print(x.solution)
print(x.is_active)