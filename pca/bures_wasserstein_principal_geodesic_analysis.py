import scipy

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from functools import reduce
from numpy.linalg import norm
from ot.gaussian import bures_wasserstein_barycenter
from scipy.linalg import inv
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
    #times_clipped = clip_times(points, mean, vec)[0]
    #if points.ndim < 3:
    #    times_clipped = [times_clipped]
    #projections_on_line = np.stack([mean + t * vec for t in times_clipped])
    #residuals = points - projections_on_line
    #sq_norms = np.squeeze(np.apply_over_axes(np.sum, residuals ** 2, [-2, -1]))
    ## sq_norms = np.stack([np.sum(res ** 2) for res in residuals])
    sq_norms = [cost_func_unit(point, mean, vec) for point in points]
    return np.sum(sq_norms)


def compute_component(points, mean, vec):
    times_clipped, n_clip = clip_times(points, mean, vec)
    projections_on_line = np.stack([mean + t * vec for t in times_clipped])
    component = project(projections_on_line)
    return component, times_clipped, n_clip


def evaluate_results(components, points_spd, mean_spd):
    dim = points_spd.shape[-1]
    space = SPDMatrices(dim)
    space.equip_with_metric(SPDBuresWassersteinMetric)
    costs = np.array([np.sum(space.metric.dist(points_spd, component) ** 2) for component in components])
    variances = np.array([np.sum(space.metric.dist(component, mean_spd) ** 2) for component in components])
    return costs, variances


class BuresWassersteinTPCA:
    def __init__(self):
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.variances = None
        self.times = None

    # def fit(self, points_spd):
    #     n_points, dim = points_spd.shape[:2]
    #     _, self.mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)
    #
    #     rdim = dim * (dim + 1) // 2
    #     metric_mat = np.zeros((rdim, rdim))
    #     basis_vectors = np.stack([from_vec_to_sym(np.eye(rdim)[:, i], dim) for i in range(rdim)])
    #     for i in range(rdim):
    #         for j in range(rdim):
    #             metric_mat[i, j] = np.trace(basis_vectors[i] @ self.mean_spd @ basis_vectors[j])
    #
    #     sym_mats = monge_map(self.mean_spd, points_spd)
    #     logs = np.stack([from_sym_to_vec(sym_mat - np.eye(dim), dim) for sym_mat in sym_mats])
    #     test_center = norm(np.sum(logs, axis=0) / n_points)
    #     if test_center > 1e-5:
    #         print(f'Warning: the norm of the mean of the tangent vectors is {test_center}, not zero.')
    #     covariance_of_logs = 1 / n_points * logs.T @ logs @ metric_mat # !!!!!!! NOT SYMMETRIC !!!!!!!
    #     print(covariance_of_logs)
    #     indices = np.argsort(np.linalg.eig(covariance_of_logs)[0])[::-1]
    #     eig_vecs = np.linalg.eig(covariance_of_logs)[1].T[indices]
    #     eig_vecs = gram_schmidt(eig_vecs, metric_mat)
    #     assert np.all(np.abs(eig_vecs @ metric_mat @ eig_vecs.T - np.eye(rdim)) < 1e-5)
    #
    #     mean = sqrtm(self.mean_spd)
    #     points = sym_mats @ mean
    #     assert np.all(np.abs(points - align(points_spd, mean)) < 1e-5)
    #
    #     #components_coords = logs @ metric_mat @ eig_vecs
    #     #components_vec = np.einsum('ji,ki->ijk', components_coords, eig_vecs) # shape (3, 11, 3)
    #     self.components = np.zeros((rdim, n_points, dim, dim))
    #     for i in range(rdim):
    #         times = logs @ metric_mat @ eig_vecs[i]
    #         component_sym_vec = np.einsum('i,j->ij', times, eig_vecs[i])
    #         component_sym = np.stack([from_vec_to_sym(vec, dim) for vec in component_sym_vec])
    #         component_vec = component_sym @ mean
    #         projections = mean + component_vec
    #         self.components[i] = project(projections)
    #
    #     vecs = np.stack([from_vec_to_sym(v, dim) for v in eig_vecs]) @ mean
    #     self.vecs_spd = np.stack([tangent_project(vec, mean) for vec in vecs])
    #     self.costs, self.variances = evaluate_results(self.components, points_spd, self.mean_spd)
    #     return self

    def fit(self, points_spd):
        n_points, dim = points_spd.shape[:2]
        _, self.mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)

        rdim = dim * (dim + 1) // 2
        metric_mat = np.zeros((rdim, rdim))
        basis_vectors = np.stack([from_vec_to_sym(np.eye(rdim)[:, i], dim) for i in range(rdim)])
        for i in range(rdim):
            for j in range(rdim):
                metric_mat[i, j] = np.trace(basis_vectors[i] @ self.mean_spd @ basis_vectors[j])

        points_sym = monge_map(self.mean_spd, points_spd)
        logs = np.stack([from_sym_to_vec(pt_sym - np.eye(dim), dim) for pt_sym in points_sym])
        test_center = norm(np.sum(logs, axis=0) / n_points)
        if test_center > 1e-5:
            print(f'Warning: the norm of the mean of the tangent vectors is {test_center}, not zero.')
        covariance_of_logs = 1 / n_points * sqrtm(metric_mat) @ logs.T @ logs @ sqrtm(metric_mat)
        eig_vecs = np.linalg.eigh(covariance_of_logs)[1][:, ::-1]
        vecs = (inv(sqrtm(metric_mat)) @ eig_vecs).T

        mean = sqrtm(self.mean_spd)
        self.components = np.zeros((rdim, n_points, dim, dim))
        self.times = np.zeros((rdim, n_points))
        for i in range(rdim):
            self.times[i] = logs @ metric_mat @ vecs[i]
            component_sym_vec = np.einsum('i,j->ij', self.times[i], vecs[i])
            component_sym = np.stack([from_vec_to_sym(vec, dim) for vec in component_sym_vec])
            component_vec = component_sym @ mean
            projections = mean + component_vec
            self.components[i] = project(projections)

        mats = np.stack([from_vec_to_sym(v, dim) for v in vecs]) @ mean
        self.vecs_spd = np.stack([tangent_project(mat, mean) for mat in mats])
        self.costs, self.variances = evaluate_results(self.components, points_spd, self.mean_spd)
        return self

    # def fit(self, points_spd):
    #     n_points, dim = points_spd.shape[:2]
    #     _, self.mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)
    #     mean = sqrtm(self.mean_spd)
    #     points = align(points_spd, mean)
    #
    #     logs = (points - mean).reshape((n_points, dim ** 2))
    #     test_center = norm(np.sum(logs, axis=0) / n_points)
    #     if test_center > 1e-5:
    #         print(f'Warning: the norm of the mean of the tangent vectors is {test_center}, not zero.')
    #     covariance_of_logs = 1 / n_points * logs.T @ logs
    #     _, eig_vecs = np.linalg.eigh(covariance_of_logs)
    #
    #     vecs = np.zeros((dim ** 2, dim, dim))
    #     self.components = np.zeros((dim ** 2, n_points, dim, dim))
    #     times = np.zeros((dim ** 2, n_points))
    #     for i in range(dim ** 2):
    #         vecs[i] = eig_vecs[:, -i-1].reshape((dim, dim))
    #         print(norm(mean.T @ vecs[i] - vecs[i].T @ mean))
    #         times[i] = np.stack([np.sum((pt - mean) * vecs[i]) for pt in points])
    #         projections_on_line = np.stack([mean + t * vecs[i] for t in times[i]])
    #         self.components[i] = project(projections_on_line)
    #
    #     #print(np.around(np.stack([vec.T @ mean - mean.T @ vec for vec in vecs]), 3))
    #     self.vecs_spd = np.stack([tangent_project(vec, mean) for vec in vecs])
    #     self.costs, self.variances = evaluate_results(self.components, points_spd, self.mean_spd)
    #     return self


class BuresWassersteinPGA:
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
            return BuresWassersteinPGA2D(**kwargs)
        else:
            return BuresWassersteinPGAND(dim, **kwargs)


class BuresWassersteinPGA2D:
    def __init__(
        self,
        max_iter=100,
        tol_1=1e-3,
        tol_2=1e-3,
    ):
        """ Principal geodesic analysis for the Bures-Wasserstein metric in dimension 2.
        """
        self.max_iter = max_iter
        self.tol_1 = tol_1
        self.tol_2 = tol_2,
        self.mean = np.zeros((3, 2, 2))
        self.vec = np.zeros((3, 2, 2))
        self.time = 0.
        self.fiber_angle = np.zeros(2)
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
            self.component_step_1(n)
            self.component_step_2(n)

            self.costs_gl[n].append(self.cost_func(n))
            if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol_1):
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
            vec_aux = np.random.rand(2, 2)
            vec = (vec_aux + vec_aux.T) @ self.mean[0]
            self.vec[0] = vec / norm(vec)

        elif n == 1:
            self.mean[1] = self.mean[0].copy()
            vec_aux = np.random.rand(2, 2)
            vec = (vec_aux + vec_aux.T) @ self.mean[1]
            vec = vec - np.sum(vec * self.vec[0]) * self.vec[0]
            self.vec[1] = vec / norm(vec)

        else:
            self.mean[2] = self.mean[1].copy()
            vec_aux = np.random.rand(2, 2)
            vec_2 = (vec_aux + vec_aux.T) @ self.mean[2]
            vec_0_mean_2 = self.vec[0] @ make_rotation_2d(self.fiber_angle[0])
            vec_1_mean_2 = self.vec[1]
            vec_2 = vec_2 - np.sum(vec_2 * vec_0_mean_2) * vec_0_mean_2 - np.sum(vec_2 * vec_1_mean_2) * vec_1_mean_2
            self.vec[2] = vec_2 / norm(vec_2)

    def component_step_1(self, n):
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

    def component_step_2(self, n):
        if n == 0:
            self.component_0_step_2()
        elif n == 1:
            self.component_1_step_2()
        else:
            self.component_2_step_2()

    def component_0_step_2(self):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal line is parametrized as mean_1 + t * vec_1, where vec_1 is a unit
        horizontal vector at mean_1. This step minimizes the cost function wrt mean_1 and
        vec_1 under both constraints (vec_1 is unit norm and horizontal) using the default
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

    def component_1_step_2(self):
        """ Step 2 of component 2: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. Therefore, it is parametrized by
        t -> mean_2 + t * vec_2, where mean_2 = (mean_1 + time * vec_1) @ rotation(angle),
        and vec_2 is a unit horizontal vector at mean_2 and is orthogonal to
        vec_1 @ rotation(angle). It is found by minimizing the cost function over angle, time,
        and vec_2 under these 3 constraints, using the default SLQSP (Sequential Least Squares
        Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x[:4].reshape((2, 2))
            fiber_angle = x[4]
            time = x[5]
            rotation = make_rotation_2d(np.squeeze(fiber_angle))
            mean = (self.mean[0] + time * self.vec[0]) @ rotation
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            a = self.mean[0].reshape(4)
            v = self.vec[0].reshape(4)
            return (
                    x[0] * (- (a[0] + x[5] * v[0]) * np.sin(x[4]) + (a[1] + x[5] * v[1]) * np.cos(x[4])) +
                    x[2] * (- (a[2] + x[5] * v[2]) * np.sin(x[4]) + (a[3] + x[5] * v[3]) * np.cos(x[4])) -
                    x[1] * ((a[0] + x[5] * v[0]) * np.cos(x[4]) + (a[1] + x[5] * v[1]) * np.sin(x[4])) -
                    x[3] * ((a[2] + x[5] * v[2]) * np.cos(x[4]) + (a[3] + x[5] * v[3]) * np.sin(x[4]))
            )

        def orthogonality_constraint(x):
            rotation_1 = make_rotation_2d(x[-2])
            vec_1_rotated = (self.vec[0] @ rotation_1).reshape(4)
            return np.sum(x[:4] * vec_1_rotated)

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
            {'type': 'eq', 'fun': orthogonality_constraint},
            {'type': 'eq', 'fun': horizontality_constraint},
        ]
        # a = mean_1.reshape(4)
        # v = vec_1.reshape(4)
        # cons = (
        #     {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
        #     {'type': 'eq', 'fun': lambda x: (
        #             (v[0] * np.cos(x[-2]) + v[1] * np.sin(x[-2])) * x[0] +
        #             (v[1] * np.cos(x[-2]) - v[0] * np.sin(x[-2])) * x[1] +
        #             (v[2] * np.cos(x[-2]) + v[3] * np.sin(x[-2])) * x[2] +
        #             (v[3] * np.cos(x[-2]) - v[2] * np.sin(x[-2])) * x[3]
        #     )},
        #     {'type': 'eq', 'fun': lambda x: (
        #             x[0] * (- (a[0] + x[-1] * v[0]) * np.sin(x[-2]) + (a[1] + x[-1] * v[1]) * np.cos(x[-2])) +
        #             x[2] * (- (a[2] + x[-1] * v[2]) * np.sin(x[-2]) + (a[3] + x[-1] * v[3]) * np.cos(x[-2])) -
        #             x[1] * ((a[0] + x[-1] * v[0]) * np.cos(x[-2]) + (a[1] + x[-1] * v[1]) * np.sin(x[-2])) -
        #             x[3] * ((a[2] + x[-1] * v[2]) * np.cos(x[-2]) + (a[3] + x[-1] * v[3]) * np.sin(x[-2]))
        #     )}
        # )

        x0 = np.hstack([self.vec[1].reshape(4), self.fiber_angle[0], self.time])
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        self.vec[1] = x_sol[:4].reshape((2, 2))
        self.fiber_angle[0] = np.squeeze(x_sol[4])
        self.time = np.squeeze(x_sol[5])
        self.mean[1] = (self.mean[0] + self.time * self.vec[0]) @ make_rotation_2d(self.fiber_angle[0])

    def component_2_step_2(self):
        """ Step 2 of component 3: find optimal horizontal line.

        The third geodesic component is parametrized by t -> mean_3 + t * vec_3, where
        mean_3 = mean_2 @ rotation(angle_2), and vec_3 is a unit horizontal vector at mean_3
        orthogonal to vec_1 @ rotation(angle_1) @ rotation(angle_2) and to vec_2 @ @ rotation(angle_2).
        It is found by minimizing the cost function over angle_2 and vec_3 under these 4 constraints,
        using the default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
        """
        def func2min(x):
            vec = x[:4].reshape((2, 2))
            fiber_angle = x[4]
            mean = self.mean[1] @ make_rotation_2d(np.squeeze(fiber_angle))
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            a = self.mean[1].reshape(4)
            return (
                    x[0] * (- a[0] * np.sin(x[4]) + a[1] * np.cos(x[4])) +
                    x[2] * (- a[2] * np.sin(x[4]) + a[3] * np.cos(x[4])) -
                    x[1] * (a[0] * np.cos(x[4]) + a[1] * np.sin(x[4])) -
                    x[3] * (a[2] * np.cos(x[4]) + a[3] * np.sin(x[4]))
            )

        def orthogonality_constraints(x):
            rotation_0 = make_rotation_2d(self.fiber_angle[0])
            rotation_1 = make_rotation_2d(x[4])
            vec_0_rotated = (self.vec[0] @ rotation_0 @ rotation_1).reshape(4)
            vec_1_rotated = (self.vec[1] @ rotation_1).reshape(4)
            return np.stack((
                np.sum(x[:4] * vec_0_rotated),
                np.sum(x[:4] * vec_1_rotated)
            ))

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
            {'type': 'eq', 'fun': orthogonality_constraints},
            {'type': 'eq', 'fun': horizontality_constraint},
        ]
        x0 = np.hstack([self.vec[2].reshape(4), self.fiber_angle[1]])
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        self.vec[2] = x_sol[:4].reshape((2, 2))
        self.fiber_angle[1] = np.squeeze(x_sol[4])
        self.mean[2] = self.mean[1] @ make_rotation_2d(self.fiber_angle[1])

    def fit(self, points_spd):
        self.points_spd = points_spd
        self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        for n in range(3):
            self.component(n)

        self.mean_spd = project(self.mean[1])
        self.vecs_spd = np.stack([
            tangent_project(self.vec[0] @ make_rotation_2d(self.fiber_angle[0]), self.mean[1]),
            tangent_project(self.vec[1], self.mean[1]),
            tangent_project(self.vec[2], self.mean[2])
        ])
        self.costs, self.variances_spd = evaluate_results(self.components, self.points_spd, self.mean_spd)

        indices = np.argsort(self.costs)
        self.vecs_spd = self.vecs_spd[indices]
        self.components = self.components[indices]
        self.costs = self.costs[indices]
        self.variances_spd = self.variances_spd[indices]
        return self


class BuresWassersteinPGAND:
    def __init__(
        self,
        dim,
        max_iter=100,
        max_iter_gd=10000,
        tol=1e-3,
        tol_1=1e-3,
        tol_2=1e-3,
        tol_gd=1e-5,
        step_size_1=0.005,
        step_size_2=0.005,
    ):
        """Principal geodesic analysis for the Bures-Wasserstein metric.
        """
        self.dim = dim
        self.max_iter = max_iter
        self.max_iter_gd = max_iter_gd
        self.tol = tol
        self.tol_1 = tol_1
        self.tol_2 = tol_2
        self.tol_gd = tol_gd
        self.step_size_1 = step_size_1
        self.step_size_2 = step_size_2
        self.n_comp = self.dim * (self.dim + 1) // 2
        self.mean_0 = np.zeros((self.dim, self.dim))
        self.mean = np.zeros((self.n_comp, self.dim, self.dim))
        self.vec = np.zeros((self.n_comp, self.dim, self.dim))
        self.fiber_rotation = np.zeros((self.n_comp - 1, self.dim, self.dim))
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
            self.component_step_1(n)
            self.component_step_2(n)

            self.costs_gl[n].append(self.cost_func(n))
            if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol_1):
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
            self.fiber_rotation[n - 1] = np.eye(self.dim)
            self.mean[n] = self.mean[n - 1].copy()
            vec_aux = np.random.rand(self.dim, self.dim)
            vec_n = (vec_aux + vec_aux.T) @ self.mean[n]
            vecs_rotated = []
            for i in range(n):
                prod_rotations = reduce(np.dot, [self.fiber_rotation[j] for j in range(i, n)])
                vecs_rotated.append(self.vec[i] @ prod_rotations)
            vecs_rotated = np.stack(vecs_rotated)
            projections = np.stack([np.sum(vec_n * vecs_rotated[i]) * vecs_rotated[i] for i in range(n)])
            vec_n = vec_n - np.sum(projections, axis=0)
            self.vec[n] = vec_n / norm(vec_n)

    def component_step_1(self, n):
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
                self.rotations[i] = SOd.metric.exp(- self.step_size_1 * riemannian_grad, old_rotation)
                self.points[i] = self.sq_roots[i] @ self.rotations[i]
                if stop_iteration(self.rotations[i], old_rotation, self.tol_gd):
                    break
                iteration += 1

            if iteration == self.max_iter_gd:
                print(f'Warning (step 1): max number {self.max_iter_gd} of iterations reached.')

    def component_step_2(self, n):
        if n == 0:
            self.component_0_step_2()
        else:
            self.component_n_step_2(n)

    def component_0_step_2(self):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal line is parametrized as mean_1 + t * vec_1, where vec_1 is a unit
        horizontal vector at mean_1. This step minimizes the cost function wrt mean_1 and
        vec_1 under both constraints (vec_1 is unit norm and horizontal) using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize.
        """
        dim = self.dim

        def func2min(x):
            vec = x[:dim ** 2].reshape((dim, dim)).T
            mean = x[dim ** 2:].reshape((dim, dim)).T
            return cost_func(self.points, mean, vec)

        # def jac(x):
        #     vec = x[:dim ** 2].reshape((dim, dim)).T
        #     mean = x[dim ** 2:].reshape((dim, dim)).T
        #     sq_norm = np.sum(vec ** 2)
        #     aux_1 = np.stack([mean - pt + (sq_norm - 2) * np.sum((mean - pt) * vec) * vec for pt in self.points])
        #     aux_2 = np.stack([
        #         np.sum((mean - pt) * vec) ** 2 * vec + (sq_norm - 2) * np.sum((mean - pt) * vec) * (mean - pt)
        #         for pt in self.points
        #     ])
        #     grad_mean = 2 * np.sum(aux_1, axis=0).T.reshape(dim ** 2)
        #     grad_vec = 2 * np.sum(aux_2, axis=0).T.reshape(dim ** 2)
        #     return np.stack((grad_vec, grad_mean))

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

    def component_n_step_2(self, n):
        """ Step 2 of component n: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. Therefore it is parametrized by
        t -> mean_2 + t * vec_2, where mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2, and
        vec_2 is a unit horizontal vector at mean_2 and is orthogonal to vec_1 @ rotation_2.
        It is found by an alternate minimization of the cost function over rotation_2 (step
        2-1) and vec_2 and time_2 (step 2-2).
        """
        self.costs_gl[n] = [self.cost_func(n)]
        iteration = 0
        while iteration < self.max_iter:
            self.component_n_step_2_1(n)
            self.component_1_step_2_2() if n == 1 else self.component_n_step_2_2(n)

            self.costs_gl[n].append(self.cost_func(n))
            if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol):
                break
            iteration += 1

        if iteration == self.max_iter:
            print('Step 2: Warning! max number of iterations reached for step 2 of component 2.')

    def component_n_step_2_1(self, n):
        """ Step 2-1 of component n: Riemannian gradient descent over fiber_rotation.
        """
        mean_to_rotate = self.mean[0] + self.time * self.vec[0] if n == 1 else self.mean[n - 1]
        iteration = 0
        while iteration < self.max_iter_gd:
            times_clipped = clip_times(self.points, self.mean[n], self.vec[n])[0]
            projections = np.stack([self.mean[n] + t * self.vec[n] for t in times_clipped])
            sum_residuals = np.sum(projections - self.points, axis=0)
            euclidean_grad = 2 * mean_to_rotate.T @ sum_residuals
            riemannian_grad = 1 / 2 * (
                    euclidean_grad - self.fiber_rotation[n-1] @ euclidean_grad.T @ self.fiber_rotation[n - 1]
            )
            SOd = SpecialOrthogonal(self.dim)
            old_fiber_rotation = self.fiber_rotation[n - 1].copy()
            self.fiber_rotation[n - 1] = SOd.metric.exp(-self.step_size_2 * riemannian_grad, self.fiber_rotation[n - 1])
            self.mean[n] = mean_to_rotate @ self.fiber_rotation[n - 1]
            self.vec[n] = self.vec[n] @ old_fiber_rotation.T @ self.fiber_rotation[n - 1]
            if stop_iteration(self.fiber_rotation[n - 1], old_fiber_rotation, self.tol):
                break
            iteration += 1

        if iteration == self.max_iter_gd:
            print(f'Warning (step 1): max number {self.max_iter_gd} of iterations reached.')

    def component_1_step_2_2(self):
        """ Step 2-2 of component 2: optimize over vec_2 and time_2.
        """
        def func2min(x):
            vec = x[:self.dim ** 2].reshape((self.dim, self.dim)).T
            time = x[-1]
            mean = (self.mean[0] + time * self.vec[0]) @ self.fiber_rotation[0]
            return cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            mean = (self.mean[0] + x[-1] * self.vec[0]) @ self.fiber_rotation[0]
            mat_cons = []
            for i in range(self.dim - 1):
                for j in range(i + 1, self.dim):
                    row = np.zeros(self.dim ** 2)
                    row[i * self.dim: (i + 1) * self.dim] = mean[:, j]
                    row[j * self.dim: (j + 1) * self.dim] = - mean[:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x[:self.dim ** 2]

        vec_0_mean_1 = (self.vec[0] @ self.fiber_rotation[0]).T.reshape(self.dim ** 2)
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
        self.mean[1] = (self.mean[0] + self.time * self.vec[0]) @ self.fiber_rotation[0]

    def component_n_step_2_2(self, n):
        """ Step 2-2 of component n: optimize over vec_n.
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
            vecs_mean_n = []
            for k in range(n):
                prod = reduce(np.dot, [self.fiber_rotation[i] for i in range(k, n)])
                vecs_mean_n.append((self.vec[k] @ prod).T.reshape(self.dim ** 2))
            vecs_mean_n = np.stack(vecs_mean_n)
            return vecs_mean_n @ x
        # def orthogonality_constraint(x, k):
        #     prod = reduce(np.dot, [self.fiber_rotation[i] for i in range(k, n)])
        #     vec_k_mean_1 = (self.vec[k] @ prod).T.reshape(self.dim ** 2)
        #     return np.sum(x * vec_k_mean_1)

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] ** 2) - 1},
            {'type': 'eq', 'fun': horizontality_constraint},
            {'type': 'eq', 'fun': orthogonality_constraints},
        ] #+ [
            #{'type': 'eq', 'fun': lambda x: orthogonality_constraint(x, k)} for k in range(n)
        #]
        x0 = self.vec[n].T.reshape(self.dim ** 2)
        h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
        x_sol = h['x']
        if not h['success']:
            print('warning: failure in step 2-2 of component', n)
        self.vec[n] = x_sol.reshape((self.dim, self.dim)).T

    # def component_3_step_2(self, rotation_3, vec_3, points, mean_2, vec_1, vec_2):
    #     """ Step 2 of component 2: find optimal horizontal line.
    #
    #     The third geodesic component is parametrized by t -> mean_3 + t * vec_3, where
    #     mean_3 = mean_2 @ rotation_3, and vec_3 is a unit horizontal vector at mean_3
    #     and is orthogonal to vec_1 @ rotation_3 and to vec_2 @ rotation_3. It is found
    #     by an alternate minimization of the cost function over rotation_3 (step 3-1) and
    #     vec_3 (step 3-2).
    #     """
    #     mean_3 = mean_2 @ rotation_3
    #     costs = [cost_func(points, mean_3, vec_3)]
    #     for iteration in range(self.max_iter):
    #         if self.super_verbose: print(f'iteration {iteration} of step 2')
    #         rotation_3 = self.component_3_step_2_1(rotation_3, vec_3, points, mean_2, vec_1, vec_2)
    #
    #         vec_3 = self.component_2_step_2_2(rotation_2, rotation_2, time_2, vec_2, points, mean_1, vec_1)
    #         mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
    #         costs.append(cost_func(points, mean_2, vec_2))
    #         if self.super_verbose: print('step 2-2, cost is ', costs[-1])
    #
    #         if np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol or np.abs(costs[-1] - costs[-2]) < self.tol:
    #             if self.verbose: print(f'Step 2: convergence reached in {iteration + 1} iterations.')
    #             break
    #
    #     if iteration == self.max_iter - 1:
    #         print('Step 2: Warning! max number of iterations reached for step 2 of component 2.')
    #     return rotation_2, time_2, vec_2

    # def evaluate_results(self, components, points_spd, mean_spd):
    #     costs = [
    #         np.sum(self.space.metric.dist(points_spd, components[0]) ** 2),
    #         np.sum(self.space.metric.dist(points_spd, components[1]) ** 2),
    #     ]
    #     variances = [
    #         np.sum(self.space.metric.dist(components[0], mean_spd) ** 2),
    #         np.sum(self.space.metric.dist(components[1], mean_spd) ** 2),
    #     ]
    #     return costs, variances

    def fit(self, points_spd):
        """ Perform Bures-Wasserstein Principal Geodesic Analysis on SPD matrices.
        """
        self.points_spd = points_spd
        self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        for n in range(self.n_comp):
            self.component(n)

        self.mean_spd = project(self.mean[1])
        self.vecs_spd = np.stack(
            [tangent_project(self.vec[0] @ self.fiber_rotation[0], self.mean[1])] +
            [tangent_project(self.vec[n], self.mean[n]) for n in range(1, self.n_comp)]
        )
        self.costs, self.variances_spd = evaluate_results(self.components, self.points_spd, self.mean_spd)

        indices = np.argsort(self.costs)
        self.vecs_spd = self.vecs_spd[indices]
        self.components = self.components[indices]
        self.costs = self.costs[indices]
        self.variances_spd = self.variances_spd[indices]

        return self
