from PIL import Image
from projection_matrices import *
from scipy_least_squares import *
from utility import *
from scipy.sparse import csgraph

class CCLS_instance:
    def __init__(self, A, y, m, z=np.zeros(0, dtype=int), mask=np.zeros(0, dtype=int)):
        self.A = A.copy()
        self.y = y.copy()
        self.m = m
        self.z = z.copy()
        self.mask = mask.copy()
        self.M, self.N = A.shape

    def fusion(self, x):
        """
        Returns the fusion of input x and the current partial solution z
        """
        new_x = x.copy()
        for i in range(self.z.size):
            new_x[self.mask[i]] = self.z[i]
        return new_x

    def compare_z(self, x):
        """
        Returns the size of the set {i in [N] : x[i] == z[i]}, where z is the current partial solution
        :param x:
        :return:
        """
        counter = 0
        for i in range(self.z.size):
            j = self.mask[i]
            if x[j] == self.z[i]:
                counter += 1
        return counter

    def problem_reduction_single(self, i, val):
        """
        Reduces the problem size by one by adding the constraint x[i] = val
        """
        y_update = - val * self.A.getcol(i).toarray().flatten()
        self.y += y_update
        self.A = sparse.hstack([self.A[:, :i], self.A[:, i + 1:]], format='csr')
        z_index = self.mask.searchsorted(i)
        self.mask = np.insert(self.mask, z_index, i)
        self.z = np.insert(self.z, z_index, val)

    def visualise_z(self, n1, n2):
        """
        Display an n1 x n2 image of the current partial solution.
        """
        full_vector = self.fusion(0.6 * np.ones(self.N))
        im = Image.fromarray(255 * full_vector.reshape((n1, n2)))
        im.show()

    def save_z_image(self, n1, n2, file_name):
        """
        Save an n1 x n2 image of the current partial solution.
        """
        full_vector = self.fusion(0.6 * np.ones(self.N))
        im = Image.fromarray(255 * full_vector.reshape((n1, n2)))
        im.convert('RGB').save(file_name)

    def reduction_size_one_autarkies(self):
        """
        Eliminate all the size one autarkies of the current problem instance
        """
        done = False
        while not done:
            current_N = self.A.shape[1]
            max_projection = self.A @ np.ones(current_N, dtype=int)
            inprods = self.A.transpose() @ max_projection
            A_y = self.A.transpose() @ self.y
            indices = []
            vals = []
            for i in range(current_N):
                if A_y[i] <= 0.5 * self.m:
                    indices.append(i)
                    vals.append(0)
                elif A_y[i] >= inprods[i] - 0.5 * self.m:
                    indices.append(i)
                    vals.append(1)
            no_reductions = len(indices)
            for i in range(no_reductions - 1, -1, -1): #reverse order
                self.problem_reduction_single(indices[i], vals[i])
            if no_reductions == 0:
                done = True

    def reduction_implication_network(self, rounding_parameter):
        """
        Use an implication network to reduce the current problem instance.
        :param rounding_parameter: the projection data will be truncated to [rounding_parameter], in order to be able
               to use integer capacities in the implication network.
        """
        current_N = self.A.shape[1]
        A_csc = self.A.tocsc()
        B = 10 ** rounding_parameter * A_csc.transpose() * A_csc
        int_y = (10 ** rounding_parameter * self.y + 0.1 * np.ones_like(self.y)).astype(int)
        diagonal = B.diagonal()
        B -= csr_matrix(np.diag(diagonal))
        A_y = self.A.transpose() @ int_y
        outeredges = csr_matrix(2 * A_y - diagonal)
        Adjacency_matrix = 2 * sparse.bmat([[None, 2 * B, None, None],
                                        [None, None, None, outeredges.transpose()],
                                        [outeredges, None, None, None],
                                        [None, None, 0, None]], format='csr')
        max_flow_output = csgraph.maximum_flow(Adjacency_matrix, 2*current_N, 2*current_N + 1)

        flow = max_flow_output.residual
        symmetric_central_flow = flow[:current_N, current_N: 2*current_N] + flow[:current_N, current_N: 2*current_N].transpose()
        symmetric_central_flow.data //= 2
        flow[:current_N, current_N: 2 * current_N] = symmetric_central_flow
        flow[current_N: 2 * current_N, :current_N] = -symmetric_central_flow
        symmetric_outer_flow = \
            flow[2*current_N, :current_N] + flow[current_N: 2 * current_N, 2*current_N + 1].transpose()
        symmetric_outer_flow.data //= 2
        flow[2 * current_N, :current_N] = symmetric_outer_flow
        flow[current_N: 2 * current_N, 2 * current_N + 1] = symmetric_outer_flow.transpose()

        residual = Adjacency_matrix - flow
        residual.eliminate_zeros()
        n_components, labels = csgraph.connected_components(residual, connection='strong')

        component_type = np.zeros(n_components, dtype=int)
        # Type 14: u and 1 - u are both contained in the component
        # Type 15: u is contained in the component, 1 - u is not, no path from u to u - 1
        # Type 16: u is contained in the component, 1 - u is not, there exists a path from u to u - 1
        indices = []
        vals = []
        for i in range(current_N):
            component = labels[i]
            if component_type[component] == 0:
                if component == labels[i + current_N]:
                    component_type[component] = 14
                else:
                    reachable = csgraph.breadth_first_order(residual, i, return_predecessors=False)
                    if i + current_N in reachable:
                        component_type[component] = 16
                    else:
                        component_type[component] = 15
            if component_type[component] == 15:
                indices.append(i)
                vals.append(1)
            elif component_type[component] == 16:
                indices.append(i)
                vals.append(0)

        no_reductions = len(indices)
        order = np.array(indices).argsort()
        for j in range(no_reductions):
            i = order[no_reductions - j - 1]
            self.problem_reduction_single(indices[i], vals[i])
        return 0

    def reduction_dual(self, tol, upperbound, x_0='default', precomputed_x_hat='not_computed'):
        """
        Use Langrangian duality to reduce the current problem instance.
        :param tol: Tolerance used for termination of the computation of $\hat x$. Setting tol = epsilon guarantees
               that $\hat x$ will be an epsilon-pseudo-optimal solution.
        :param upperbound: Upper bound on the optimal objective value of BCLS. If such an upper bound is not known enter
                           "'not_computed'".
        :param x_0: Initial guess for an optimal solution of the relaxed discrete tomography problem (CCLS). Enter
                    "'default'" to use x0[i] = 0.5 for each i.
        :param precomputed_x_hat: In case an (almost) optimal solution of CCLS was already computed, enter it here.
               Otherwise an tol-pseudo-optimal solution will be computed by using a least squares solver.
        """
        current_N = self.A.shape[1]
        if precomputed_x_hat is 'not_computed':
            if x_0 is 'default':
                x_0 = 0.5 * np.ones(current_N)
            x_hat = compute_constrained_least_squares_solution(self.A, self.y, x_0, tol).x
        else:
            x_hat = precomputed_x_hat
        nu_hat = self.A.transpose() @ (self.y - self.A @ x_hat)
        relaxed_cost = 0.5 * np.linalg.norm(self.A @ x_hat - self.y) ** 2
        L = np.zeros(current_N)
        indices = []
        vals = []
        for i in range(current_N):
            L[i] = abs(nu_hat[i]) - self.m * np.sqrt(2 * self.M * current_N * tol)
            if L[i] > 0:
                if upperbound < relaxed_cost + L[i] ** 2 / relaxed_cost + L[i] - current_N * tol:
                    indices.append(i)
                    if nu_hat[i] < 0:
                        vals.append(0)
                    elif nu_hat[i] > 0:
                        vals.append(1)
        no_reductions = len(indices)
        for i in range(no_reductions - 1, -1, -1):  # reverse order
            self.problem_reduction_single(indices[i], vals[i])
