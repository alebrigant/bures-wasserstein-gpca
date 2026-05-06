import scipy

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from numpy.linalg import norm
from tools.compute import *


def stop_iteration(new_value, old_value, tol):
    criterion_1 = norm(new_value - old_value) / norm(old_value) < tol
    criterion_2 = norm(new_value - old_value) < tol
    return criterion_1 or criterion_2


def clip_times(points, mean, vec):
    if points.ndim < 3:
        times = np.sum((points - mean) * vec)
    else:
        times = np.stack([np.sum((pt - mean) * vec) for pt in points])
    eigval, _ = np.linalg.eigh(vec @ np.linalg.inv(mean))
    lbd_min, lbd_max = eigval[0], eigval[-1]
    time_inf = - 1 / lbd_max if lbd_max > 0 else -np.infty
    time_sup = - 1 / lbd_min if lbd_min < 0 else np.infty
    times_clipped = np.minimum(times, time_sup)
    times_clipped = np.maximum(times_clipped, time_inf)
    n_clip = np.sum(times != times_clipped)
    return times_clipped, n_clip


def cost_func_unit(point, mean, vec):
    time_clipped = clip_times(point, mean, vec)[0]
    residual = point - (mean + time_clipped * vec)
    return np.sum(residual ** 2)


def cost_func(points, mean, vec):
    # Sum of squared norms of the residuals of the projections of points
    # on the line going through mean, directed by vec.
    sq_norms = [cost_func_unit(point, mean, vec) for point in points]
    return np.sum(sq_norms)


def compute_component(points, mean, vec):
    times_clipped, n_clip = clip_times(points, mean, vec)
    projections_on_line = np.stack([mean + t * vec for t in times_clipped])
    component = project(projections_on_line)
    return component, times_clipped, n_clip


class BuresWassersteinGPCA:
    """ Principal geodesic analysis for the Bures-Wasserstein metric.

    This class implements principal geodesic analysis using the Otto-Wasserstein fiber
    bundle structure with base space the space of centered Gaussian distributions,
    (SPD matrices) and total space, the space of linear transformations (invertible
    matrices). The algorithm searches for a horizontal line in the total space that
    minimizes a cost function, given by the sum of squared residuals of the projections
    of the fiber representatives.

    Each component is found by an iterative procedure, initialized by:
    - a choice of fiber representatives of the SPD matrices
    - and a horizontal line in the total space of invertible matrices
    Then we perform an alternate minimization of the cost function in 2 steps:
    step 1: given the horizontal line, minimize w.r.t. the fiber representatives
            (parametrized by rotation matrices)
    step 2: given the fiber representatives, minimize w.r.t. the horizontal line
            (parametrized by mean and directing horizontal vector).
    """
    def __new__(cls, dim, **kwargs):
        if dim == 2:
            return BuresWassersteinGPCA2D(**kwargs)
        else:
            return BuresWassersteinGPCAND(dim, **kwargs)


class BuresWassersteinGPCA2D:
    def __init__(
        self,
        max_iter=100,
        tol=1e-3,
        sym_matrix_init=None,
    ):
        """ Principal geodesic analysis for the Bures-Wasserstein metric in dimension 2.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.sym_matrix_init = sym_matrix_init
        self.mean = np.zeros((3, 2, 2))
        self.vec = np.zeros((3, 2, 2))
        self.time = 0.
        self.costs_gl = {}
        self.n_clip = np.zeros(3)
        self.components = None
        self.angles = None
        self.sq_roots = None
        self.points = None
        self.points_spd = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.variances_spd = None
        self.times = None

    def cost_func(self, n):
        """Cost function of component n_component.
        """
        return cost_func(self.points, self.mean[n], self.vec[n])

    def component(self, n):
        """ First geodesic component.
        """
        self.component_init(n)
        self.costs_gl[n] = [self.cost_func(n)]

        iteration = 0
        while iteration < self.max_iter:
            self.step_1_of_component(n)
            self.step_2_of_component(n)

            self.costs_gl[n].append(self.cost_func(n))
            if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol):
                break
            iteration += 1

        if iteration == self.max_iter:
            print(f'Warning: max number of iterations reached for component {n+1}.')

        self.components[n], self.times[n], self.n_clip[n] = compute_component(self.points, self.mean[n], self.vec[n])

    def component_init(self, n):
        """ Initialization of the component and corresponding fiber representatives.

        - Component 1: initialize fiber representatives with square roots, the mean with
        their Euclidean mean, and randomly initialize a horizontal vector at mean.
        - Component 2: the fiber representatives and mean are those found for component 1,
        and initialize a random unit horizontal vector at that mean that is orthogonal to vec_1.
        - Component 3: the fiber representatives and mean are those found for component 2,
        and initialize a random unit horizontal vector at that mean that is orthogonal to vec_1
        and vec_2.
        """
        if n == 0:
            n_points = self.sq_roots.shape[0]
            self.components = np.zeros((3, n_points, 2, 2))
            self.times = np.zeros((3, n_points))
            self.angles = np.zeros(n_points)
            self.points = points_from_angles_2d(self.angles, self.sq_roots)
            self.mean[0] = np.sum(self.points, axis=0) / n_points
            if self.sym_matrix_init is None:
                vec_aux = np.random.rand(2, 2)
                self.sym_matrix_init = vec_aux + vec_aux.T
            vec = self.sym_matrix_init @ self.mean[0]
            self.vec[0] = vec / norm(vec)

        elif n == 1:
            self.mean[1] = self.mean[0].copy()
            vec = self.sym_matrix_init @ self.mean[1]
            vec = vec - np.sum(vec * self.vec[0]) * self.vec[0]
            self.vec[1] = vec / norm(vec)

        else:
            self.mean[2] = self.mean[1].copy()
            vec_2 = self.sym_matrix_init @ self.mean[2]
            vec_2 = vec_2 - np.sum(vec_2 * self.vec[0]) * self.vec[0] - np.sum(vec_2 * self.vec[1]) * self.vec[1]
            self.vec[2] = vec_2 / norm(vec_2)

    def step_1_of_component(self, n):
        """ Step 1 of component k: find optimal fiber representatives.

        The optimal representatives are parametrized as rotations(angles) @ sq_roots.
        The optimal angles are found by minimizing the cost function with the default
        BFGS method of scipy minimize.
        """
        def func2min(x):
            points = points_from_angles_2d(x, self.sq_roots)
            return cost_func(points, self.mean[n], self.vec[n])

        h = scipy.optimize.minimize(func2min, x0=self.angles)
        self.angles = h['x']
        self.points = points_from_angles_2d(self.angles, self.sq_roots)

    def step_2_of_component(self, n):
        if n == 0:
            self.step_2_of_component_0()
        elif n == 1:
            self.step_2_of_component_1()
        else:
            self.step_2_of_component_2()

    def step_2_of_component_0(self):
        """ Step 2 of component 0: find optimal horizontal line.

        The horizontal line is parametrized as mean_0 + t * vec_0, where vec_0 is a unit
        horizontal vector at mean_0. This step minimizes the cost function wrt mean_0 and
        vec_0 under both constraints (vec_0 is unit norm and horizontal) using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize. In the two-
        dimensional case, the horizontality condition reduces to one scalar equation.
        """
        def func2min(x):
            vec = x[:4].reshape((2, 2))
            mean = x[4:].reshape((2, 2))
            return cost_func(self.points, mean, vec)

        cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},  # unit norm
        {'type': 'eq', 'fun': lambda x: x[0] * x[5] - x[1] * x[4] + x[2] * x[7] - x[3] * x[6]}
        ]
        x0 = np.hstack([self.vec[0].reshape(4), self.mean[0].reshape(4)])
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        self.vec[0] = x_sol[:4].reshape((2, 2))
        self.mean[0] = x_sol[4:].reshape((2, 2))

    def step_2_of_component_1(self):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. For simplicity, we impose that these
        two lifted components (i.e. horizontal straight lines in GL(2)) intersect. Therefore,
        the lift of the second component is paramatrized by t -> mean_1 + t * vec_1, where
        mean_1 = mean_0 + time * vec_0, under the constraints:
        (1) vec_1 is a horizontal vector at mean_1
        (2) vec_1 has unit norm
        (3) vec_1 is orthogonal to vec_0.
        This step minimizes the cost function over variables time and vec_2 under these 3 constraints,
        using the default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x[:4].reshape((2, 2))
            time = x[4]
            mean = (self.mean[0] + time * self.vec[0])
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            a = self.mean[0].reshape(4)
            v = self.vec[0].reshape(4)
            return (
                    x[0] * (a[1] + x[4] * v[1]) -
                    x[1] * (a[0] + x[4] * v[0]) +
                    x[2] * (a[3] + x[4] * v[3]) -
                    x[3] * (a[2] + x[4] * v[2])
            )

        def orthogonality_constraint(x):
            return np.sum(x[:4] * self.vec[0].reshape(4))

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
            {'type': 'eq', 'fun': orthogonality_constraint},
            {'type': 'eq', 'fun': horizontality_constraint},
        ]
        x0 = np.hstack([self.vec[1].reshape(4), self.time])
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        self.vec[1] = x_sol[:4].reshape((2, 2))
        self.time = np.squeeze(x_sol[4])
        self.mean[1] = self.mean[0] + self.time * self.vec[0]

    def step_2_of_component_2(self):
        """ Step 2 of component 2: find optimal horizontal line.

        The last geodesic component is parametrized by t -> mean_2 + t * vec_2, where
        mean_2 = mean_1 and
        (1) vec_2 is a horizontal vector at mean_2
        (2) vec_2 has unit norm
        (3) vec_2 is orthogonal to vec_1 and to vec_0.
        This step minimizes the cost function over vec_2 under these 3 constraints, using the
        default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x[:4].reshape((2, 2))
            return cost_func(self.points, self.mean[1], vec)

        def horizontality_constraint(x):
            a = self.mean[1].reshape(4)
            return x[0] * a[1] - x[1] * a[0] + x[2] * a[3] - x[3] * a[2]

        def orthogonality_constraints(x):
            return np.stack((
                np.sum(x[:4] * self.vec[0].reshape(4)),
                np.sum(x[:4] * self.vec[1].reshape(4))
            ))

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
            {'type': 'eq', 'fun': orthogonality_constraints},
            {'type': 'eq', 'fun': horizontality_constraint},
        ]
        x0 = self.vec[2].reshape(4)
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        self.vec[2] = x_sol[:4].reshape((2, 2))

    def fit(self, points_spd):
        self.points_spd = points_spd
        self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        for n in range(3):
            self.component(n)

        self.mean_spd = project(self.mean[1])
        self.vecs_spd = np.stack([
            tangent_project(self.vec[0], self.mean[1]),
            tangent_project(self.vec[1], self.mean[1]),
            tangent_project(self.vec[2], self.mean[1])
        ])
        self.costs, self.variances_spd = compute_costs_and_variances(self.components, self.points_spd, self.mean_spd)

        indices = np.argsort(self.costs)
        self.vecs_spd = self.vecs_spd[indices]
        self.components = self.components[indices]
        self.costs = self.costs[indices]
        self.variances_spd = self.variances_spd[indices]
        return self


class BuresWassersteinGPCAND:
    def __init__(
        self,
        dim,
        max_iter=100,
        max_iter_gd=10000,
        tol=1e-3,
        tol_gd=1e-5,
        step_size=0.005,
    ):
        """Principal geodesic analysis for the Bures-Wasserstein metric.
        """
        self.dim = dim
        self.max_iter = max_iter
        self.max_iter_gd = max_iter_gd
        self.tol = tol
        self.tol_gd = tol_gd
        self.step_size = step_size
        self.n_comp = self.dim * (self.dim + 1) // 2
        self.mean_0 = np.zeros((self.dim, self.dim))
        self.mean = np.zeros((self.n_comp, self.dim, self.dim))
        self.vec = np.zeros((self.n_comp, self.dim, self.dim))
        self.time = 0.
        self.costs_gl = {}
        self.n_clip = np.zeros(self.n_comp)
        self.components = None
        self.rotations = None
        self.sq_roots = None
        self.points = None
        self.points_spd = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.variances_spd = None
        self.times = None

    def cost_func(self, n):
        """Cost function of component n_component.
        """
        return cost_func(self.points, self.mean[n], self.vec[n])

    def component(self, n):
        self.component_init(n)
        self.costs_gl[n] = [self.cost_func(n)]

        iteration = 0
        while iteration < self.max_iter:
            self.step_1_of_component(n)
            self.step_2_of_component(n)

            self.costs_gl[n].append(self.cost_func(n))
            if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol):
                break
            iteration += 1

        if iteration == self.max_iter:
            print(f'Warning: max number of iterations reached for component {n + 1}.')

        self.components[n], self.times[n], self.n_clip[n] = compute_component(self.points, self.mean[n], self.vec[n])

    def component_init(self, n):
        if n == 0:
            n_points = self.sq_roots.shape[0]
            self.components = np.zeros((self.n_comp, n_points, self.dim, self.dim))
            self.times = np.zeros((self.n_comp, n_points))
            self.rotations = np.tile(np.eye(self.dim), (n_points, 1, 1))
            self.points = self.sq_roots @ self.rotations
            self.mean[0] = np.sum(self.points, axis=0) / n_points
            vec_aux = np.random.rand(self.dim, self.dim)
            vec_1 = (vec_aux + vec_aux.T) @ self.mean[0]
            self.vec[0] = vec_1 / norm(vec_1)

        else:
            self.mean[n] = self.mean[n - 1].copy()
            vec_aux = np.random.rand(self.dim, self.dim)
            vec_n = (vec_aux + vec_aux.T) @ self.mean[n]
            projections = np.stack([np.sum(vec_n * self.vec[i]) * self.vec[i] for i in range(n)])
            vec_n = vec_n - np.sum(projections, axis=0)
            self.vec[n] = vec_n / norm(vec_n)

    def step_1_of_component(self, n):
        """ Step 1 of component n: find optimal fiber representatives.

        The optimal representatives are parametrized as sq_roots @ rotations. The optimal
        rotations are found by Riemannian gradient descent on the cost function in the
        space of orthogonal matrices.
        """
        n_points = self.rotations.shape[0]
        for i in range(n_points):
            iteration = 0
            while iteration < self.max_iter_gd:
                time_clipped = clip_times(self.points[i], self.mean[n], self.vec[n])[0]
                residual = self.mean[n] + time_clipped * self.vec[n] - self.sq_roots[i] @ self.rotations[i]
                euclidean_grad = - 2 * self.sq_roots[i].T @ residual
                riemannian_grad = 1 / 2 * (euclidean_grad - self.rotations[i] @ euclidean_grad.T @ self.rotations[i])
                SOd = SpecialOrthogonal(self.dim)
                old_rotation = self.rotations[i].copy()
                self.rotations[i] = SOd.metric.exp(- self.step_size * riemannian_grad, old_rotation)
                self.points[i] = self.sq_roots[i] @ self.rotations[i]
                if stop_iteration(self.rotations[i], old_rotation, self.tol_gd):
                    break
                iteration += 1

            if iteration == self.max_iter_gd:
                print(f'Warning (step 1): max number {self.max_iter_gd} of iterations reached.')

    def step_2_of_component(self, n):
        if n == 0:
            self.step_2_of_component_0()
        elif n == 1:
            self.step_2_of_component_1()
        else:
            self.step_2_of_component_n(n)

    def step_2_of_component_0(self):
        """ Step 2 of component 0: find optimal horizontal line.

        The horizontal line is parametrized as mean_0 + t * vec_0, where vec_0 is a unit
        horizontal vector at mean_0. This step minimizes the cost function wrt mean_0 and
        vec_0 under both constraints (vec_0 is unit norm and horizontal) using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize.
        """
        dim = self.dim

        def func2min(x):
            vec = x[:dim ** 2].reshape((dim, dim)).T
            mean = x[dim ** 2:].reshape((dim, dim)).T
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            mean = x[dim ** 2:].reshape((dim, dim)).T
            mat_cons = []
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    row = np.zeros(dim ** 2)
                    row[i * dim: (i + 1) * dim] = mean[:, j]
                    row[j * dim: (j + 1) * dim] = - mean[:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x[:dim ** 2]

        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x[:dim ** 2] ** 2) - 1},
            {'type': 'eq', 'fun': horizontality_constraint},
        )
        x0 = np.hstack((self.vec[0].T.reshape(dim * dim), self.mean[0].T.reshape(dim * dim)))
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons) #, jac=jac) #, options={'disp': True}) #,method='trust-constr'
        x_sol = h['x']
        if not h['success']:
            print('warning: failure in step 2 of component 0')
        self.vec[0] = x_sol[:dim ** 2].reshape((dim, dim)).T
        self.mean[0] = x_sol[dim ** 2:].reshape((dim, dim)).T

    def step_2_of_component_1(self):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. For simplicity, we impose that these
        two lifted components (i.e. horizontal straight lines in GL(2)) intersect. Therefore,
        the lift of the second component is paramatrized by t -> mean_1 + t * vec_1, where
        mean_1 = mean_0 + time * vec_0, under the constraints:
        (1) vec_1 is a horizontal vector at mean_1
        (2) vec_1 has unit norm
        (3) vec_1 is orthogonal to vec_0.
        This step minimizes the cost function over variables time and vec_1 under these 3 constraints,
        using the default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x[:self.dim ** 2].reshape((self.dim, self.dim)).T
            time = x[-1]
            mean = (self.mean[0] + time * self.vec[0])
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            mean = (self.mean[0] + x[-1] * self.vec[0])
            mat_cons = []
            for i in range(self.dim - 1):
                for j in range(i + 1, self.dim):
                    row = np.zeros(self.dim ** 2)
                    row[i * self.dim: (i + 1) * self.dim] = mean[:, j]
                    row[j * self.dim: (j + 1) * self.dim] = - mean[:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x[:self.dim ** 2]

        vec_0_mean_1 = self.vec[0].T.reshape(self.dim ** 2)
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] ** 2) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] * vec_0_mean_1)},
            {'type': 'eq', 'fun': horizontality_constraint},
        ]
        x0 = np.hstack([self.vec[1].T.reshape(self.dim ** 2), [self.time]])
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        if not h['success']:
            print('warning: failure in step 2-2 of component 1')
        self.vec[1] = x_sol[:self.dim ** 2].reshape((self.dim, self.dim)).T
        self.time = x_sol[-1]
        self.mean[1] = self.mean[0] + self.time * self.vec[0]

    def step_2_of_component_n(self, n):
        """ Step 2 of component n (n>=2): find optimal horizontal line.

        The following geodesic components are parametrized by t -> mean_n + t * vec_n, where
        mean_n = mean_1 and
        (1) vec_n is a horizontal vector at mean_n
        (2) vec_n has unit norm
        (3...) vec_n is orthogonal to vec_{n-1}, ..., vec_0.
        This step minimizes the cost function over vec_n under these constraints, using the
        default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x.reshape((self.dim, self.dim)).T
            return cost_func(self.points, self.mean[n], vec)

        def horizontality_constraint(x):
            mat_cons = []
            for i in range(self.dim - 1):
                for j in range(i + 1, self.dim):
                    row = np.zeros(self.dim ** 2)
                    row[i * self.dim: (i + 1) * self.dim] = self.mean[n][:, j]
                    row[j * self.dim: (j + 1) * self.dim] = - self.mean[n][:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x

        def orthogonality_constraints(x):
            vecs = np.stack([self.vec[k].T.reshape(self.dim ** 2) for k in range(n)])
            return vecs @ x

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] ** 2) - 1},
            {'type': 'eq', 'fun': horizontality_constraint},
            {'type': 'eq', 'fun': orthogonality_constraints},
        ]
        x0 = self.vec[n].T.reshape(self.dim ** 2)
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        if not h['success']:
            print('warning: failure in step 2-2 of component', n)
        self.vec[n] = x_sol.reshape((self.dim, self.dim)).T

    def fit(self, points_spd):
        """ Perform Bures-Wasserstein Principal Geodesic Analysis on SPD matrices.
        """
        self.points_spd = points_spd
        self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        for n in range(self.n_comp):
            self.component(n)

        self.mean_spd = project(self.mean[1])
        self.vecs_spd = np.stack(
            [tangent_project(self.vec[0], self.mean[1])] +
            [tangent_project(self.vec[n], self.mean[n]) for n in range(1, self.n_comp)]
        )
        self.costs, self.variances_spd = compute_costs_and_variances(self.components, self.points_spd, self.mean_spd)

        indices = np.argsort(self.costs)
        self.vecs_spd = self.vecs_spd[indices]
        self.components = self.components[indices]
        self.costs = self.costs[indices]
        self.variances_spd = self.variances_spd[indices]

        return self
