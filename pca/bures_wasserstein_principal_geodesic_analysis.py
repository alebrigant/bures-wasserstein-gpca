import scipy

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from functools import reduce
from numpy.linalg import norm
from ot.gaussian import bures_wasserstein_barycenter
from scipy.linalg import inv, logm, expm
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


def evaluate_results(components, points_spd, mean_spd):
    dim = points_spd.shape[-1]
    space = SPDMatrices(dim)
    space.equip_with_metric(SPDBuresWassersteinMetric)
    costs = np.array([np.sum(space.metric.dist(points_spd, component) ** 2) for component in components])
    variances = np.array([np.sum(space.metric.dist(component, mean_spd) ** 2) for component in components])
    return costs, variances


def weighted_pca_on_symmetric_matrices(points_sym, spd_mat):
    n_points, dim = points_sym.shape[:2]
    rdim = dim * (dim + 1) // 2
    metric_mat = np.zeros((rdim, rdim))
    basis_vectors = np.stack([from_vec_to_sym(np.eye(rdim)[:, i], dim) for i in range(rdim)])
    for i in range(rdim):
        for j in range(rdim):
            metric_mat[i, j] = np.trace(basis_vectors[i] @ spd_mat @ basis_vectors[j])

    points = np.stack([from_sym_to_vec(pt_sym, dim) for pt_sym in points_sym])
    mean = np.mean(points, axis=0)
    mean_sym = from_vec_to_sym(mean, dim)
    points = points - mean
    covariance = 1 / n_points * sqrtm(metric_mat) @ points.T @ points @ sqrtm(metric_mat)
    eigenvecs = np.linalg.eigh(covariance)[1][:, ::-1]
    vecs = (inv(sqrtm(metric_mat)) @ eigenvecs).T
    times = np.stack([points @ metric_mat @ vec for vec in vecs])
    components = mean + np.einsum('ki,kj->kij', times, vecs)
    vecs_sym = np.stack([from_vec_to_sym(v, dim) for v in vecs])
    components_sym = np.zeros((rdim, n_points, dim, dim))
    for i in range(rdim):
        components_sym[i] = np.stack([from_vec_to_sym(vec, dim) for vec in components[i]])

    return components_sym, times, vecs_sym, mean_sym

# def log_euclidean_pca(points_spd):
#     n_points, dim = points_spd.shape[:2]
#     rdim = dim * (dim + 1) // 2
#     logs = np.stack([logm(pt) for pt in points_spd])
#     mean_spd = expm(np.mean(logs, axis=0))
#     components_sym, times, vecs_sym = weighted_pca_on_symmetric_matrices(logs, np.eye(dim))
#     components_spd = np.zeros((rdim, n_points, dim, dim))
#     for i in range(rdim):
#         components_spd[i] = np.stack([expm(mat) for mat in components_sym[i]])
#     vecs_spd = np.stack([expm(vec) for vec in vecs_sym])
#     return components_spd, vecs_spd, mean_spd, times


def from_spd_to_cholesky_vec(mat):
    dim = mat.shape[0]
    chol = np.linalg.cholesky(mat)
    return chol.T[np.triu_indices(dim)]


def from_cholesky_vec_to_spd(vec, dim):
    mat = np.zeros((dim, dim))
    mat[np.triu_indices(dim)] = vec
    chol = mat.T
    return chol @ chol.T


class EuclideanPCA:
    def __init__(self):
        self.components = None
        self.mean = None
        self.vecs = None
        self.costs = None
        self.times = None

    def fit(self, points):
        n_points, dim = points.shape
        self.mean = np.sum(points, 0) / n_points
        points = points - self.mean
        covariance = 1 / n_points * points.T @ points
        self.vecs = np.linalg.eigh(covariance)[1][:, ::-1]
        self.times = (points @ self.vecs).T
        self.components = self.mean + np.einsum('ki,jk->kij', self.times, self.vecs)
        return self


class SPDEuclideanPCA:
    def __init__(self):
        self.points_spd = None
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.times = None

    def fit(self, points_spd):
        self.points_spd = points_spd
        n_points, dim = points_spd.shape[:2]
        points = points_spd.reshape((n_points, dim ** 2))
        # mean = np.sum(points, 0) / n_points
        # points = points - mean
        # covariance = 1 / n_points * points.T @ points
        # vecs = np.linalg.eigh(covariance)[1][:, ::-1]
        # times = (points @ vecs).T
        # components = mean + np.einsum('ki,kj->kij', times, vecs)
        print(points.shape)
        res_pca = EuclideanPCA().fit(points)
        self.times = res_pca.times
        self.components = res_pca.components.reshape((dim ** 2, n_points, dim, dim))
        self.mean_spd = res_pca.mean.reshape((dim, dim))
        self.vecs_spd = res_pca.vecs.reshape((dim ** 2, dim, dim))
        return self


class CholeskyPCA:
    def __init__(self):
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.times = None

    def fit(self, points_spd):
        n_points, dim = points_spd.shape[:2]
        points = np.stack([from_spd_to_cholesky_vec(mat) for mat in points_spd])
        # mean = np.sum(points, 0) / n_points
        # points = points - mean
        # covariance = 1 / n_points * points.T @ points
        # vecs = np.linalg.eigh(covariance)[1][:, ::-1]
        # times = (points @ vecs).T
        # components = mean + np.einsum('ki,kj->kij', times, vecs)
        res_pca = EuclideanPCA().fit(points)
        n_comp = dim * (dim + 1) // 2
        self.components = []
        for i in range(n_comp):
            comp_i = res_pca.components[i]
            self.components.append(np.stack([from_cholesky_vec_to_spd(vec, dim) for vec in comp_i]))
        self.components = np.stack(self.components)
        self.times = res_pca.times
        self.mean_spd = from_cholesky_vec_to_spd(res_pca.mean, dim)
        self.vecs_spd = np.stack([from_cholesky_vec_to_spd(vec, dim) for vec in res_pca.vecs])
        return self


class LogEuclideanGPCA:
    def __init__(self):
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.times = None

    def fit(self, points_spd):
        n_points, dim = points_spd.shape[:2]
        logs = np.stack([logm(pt) for pt in points_spd])
        self.mean_spd = expm(np.mean(logs, axis=0))
        components_sym, times, vecs_sym, mean_sym = weighted_pca_on_symmetric_matrices(logs, np.eye(dim))
        self.times = times
        self.vecs_spd = np.stack([logm_differential(mean_sym, vec_sym) for vec_sym in vecs_sym])
        # rdim = dim ** 2
        # self.mean_spd = expm(np.mean(logs, axis=0))
        # points = np.reshape(logs, (n_points, rdim))
        # mean = np.mean(points, axis=0)
        # points = points - mean
        # covariance = 1 / n_points * points.T @ points
        # eigenvecs = np.linalg.eigh(covariance)[1][:, ::-1]
        # print(eigenvecs.reshape((rdim, dim, dim)))
        # times = np.stack([points @ vec for vec in eigenvecs])
        # components = mean + np.einsum('ki,kj->kij', times, eigenvecs)
        # components_mat = np.stack([np.reshape(comp, (n_points, dim, dim)) for comp in components])
        # eigenvecs_mat = np.reshape(eigenvecs, (rdim, dim, dim))
        rdim = dim * (dim + 1) // 2
        self.components = np.zeros((rdim, n_points, dim, dim))
        for i in range(rdim):
            self.components[i] = np.stack([expm(mat) for mat in components_sym[i]])
        return self


class BuresWassersteinTPCAnew:
    def __init__(self):
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.variances = None
        self.times = None

    def fit(self, points_spd):
        n_points, dim = points_spd.shape[:2]
        _, self.mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)

        points_sym = monge_map(self.mean_spd, points_spd) - np.tile(np.eye(dim), (n_points, 1, 1))
        components_sym, self.times, vecs_sym, _ = weighted_pca_on_symmetric_matrices(points_sym, self.mean_spd)

        mean = sqrtm(self.mean_spd)
        components_gl = mean + components_sym @ mean
        self.components = np.stack([project(comp) for comp in components_gl])
        self.vecs_spd = np.stack([tangent_project(vec @ mean, mean) for vec in vecs_sym])
        self.costs, self.variances = evaluate_results(self.components, points_spd, self.mean_spd)
        return self


class BuresWassersteinTPCA:
    def __init__(self):
        self.points_spd = None
        self.components = None
        self.mean_spd = None
        self.vecs_spd = None
        self.costs = None
        self.variances = None
        self.times = None

    def fit(self, points_spd):
        self.points_spd = points_spd
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
        self.costs, self.variances_spd = evaluate_results(self.components, self.points_spd, self.mean_spd)

        indices = np.argsort(self.costs)
        self.vecs_spd = self.vecs_spd[indices]
        self.components = self.components[indices]
        self.costs = self.costs[indices]
        self.variances_spd = self.variances_spd[indices]

        return self


# class BuresWassersteinPGAND:
#     def __init__(
#         self,
#         dim,
#         max_iter=100,
#         max_iter_gd=10000,
#         tol=1e-3,
#         tol_gd=1e-5,
#         step_size=0.005,
#         sym_matrix_init=None,
#     ):
#         """Principal geodesic analysis for the Bures-Wasserstein metric.
#         """
#         self.dim = dim
#         self.max_iter = max_iter
#         self.max_iter_gd = max_iter_gd
#         self.tol = tol
#         self.tol_gd = tol_gd
#         self.step_size = step_size
#         self.sym_matrix_init = sym_matrix_init
#         self.n_comp = self.dim * (self.dim + 1) // 2
#         self.mean_0 = np.zeros((self.dim, self.dim))
#         self.mean = np.zeros((self.n_comp, self.dim, self.dim))
#         self.vec = np.zeros((self.n_comp, self.dim, self.dim))
#         self.time = 0.
#         self.costs_gl = {}
#         self.n_clip = np.zeros(self.n_comp)
#         self.components = None
#         self.rotations = None
#         self.sq_roots = None
#         self.points = None
#         self.points_spd = None
#         self.mean_spd = None
#         self.vecs_spd = None
#         self.costs = None
#         self.variances_spd = None
#         self.times = None
#
#     def cost_func(self, n):
#         """Cost function of component n_component.
#         """
#         return cost_func(self.points, self.mean[n], self.vec[n])
#
#     def component(self, n):
#         self.component_init(n)
#         self.costs_gl[n] = [self.cost_func(n)]
#
#         iteration = 0
#         while iteration < self.max_iter:
#             self.step_1_of_component(n)
#             self.step_2_of_component(n)
#
#             self.costs_gl[n].append(self.cost_func(n))
#             if stop_iteration(self.costs_gl[n][-1], self.costs_gl[n][-2], self.tol):
#                 break
#             iteration += 1
#
#         if iteration == self.max_iter:
#             print(f'Warning: max number of iterations reached for component {n + 1}.')
#
#         self.components[n], self.times[n], self.n_clip[n] = compute_component(self.points, self.mean[n], self.vec[n])
#
#     def component_init(self, n):
#         if n == 0:
#             n_points = self.sq_roots.shape[0]
#             self.components = np.zeros((self.n_comp, n_points, self.dim, self.dim))
#             self.times = np.zeros((self.n_comp, n_points))
#             self.rotations = np.tile(np.eye(self.dim), (n_points, 1, 1))
#             self.points = self.sq_roots @ self.rotations
#             self.mean[0] = np.sum(self.points, axis=0) / n_points
#             if self.sym_matrix_init is None:
#                 vec_aux = np.random.rand(self.dim, self.dim)
#                 self.sym_matrix_init = vec_aux + vec_aux.T
#             vec_0 = self.sym_matrix_init @ self.mean[0]
#             self.vec[0] = vec_0 / norm(vec_0)
#
#         else:
#             self.mean[n] = self.mean[n - 1].copy()
#             vec_aux = np.random.rand(self.dim, self.dim)
#             vec_n = (vec_aux + vec_aux.T) @ self.mean[n]
#             projections = np.stack([np.sum(vec_n * self.vec[i]) * self.vec[i] for i in range(n)])
#             vec_n = vec_n - np.sum(projections, axis=0)
#             self.vec[n] = vec_n / norm(vec_n)
#
#     def step_1_of_component(self, n):
#         """ Step 1 of component n: find optimal fiber representatives.
#
#         The optimal representatives are parametrized as sq_roots @ rotations. The optimal
#         rotations are found by Riemannian gradient descent on the cost function in the
#         space of orthogonal matrices.
#         """
#         n_points = self.rotations.shape[0]
#         for i in range(n_points):
#             iteration = 0
#             while iteration < self.max_iter_gd:
#                 time_clipped = clip_times(self.points[i], self.mean[n], self.vec[n])[0]
#                 residual = self.mean[n] + time_clipped * self.vec[n] - self.sq_roots[i] @ self.rotations[i]
#                 euclidean_grad = - 2 * self.sq_roots[i].T @ residual
#                 riemannian_grad = 1 / 2 * (euclidean_grad - self.rotations[i] @ euclidean_grad.T @ self.rotations[i])
#                 SOd = SpecialOrthogonal(self.dim)
#                 old_rotation = self.rotations[i].copy()
#                 self.rotations[i] = SOd.metric.exp(- self.step_size * riemannian_grad, old_rotation)
#                 self.points[i] = self.sq_roots[i] @ self.rotations[i]
#                 if stop_iteration(self.rotations[i], old_rotation, self.tol_gd):
#                     break
#                 iteration += 1
#
#             if iteration == self.max_iter_gd:
#                 print(f'Warning (step 1): max number {self.max_iter_gd} of iterations reached.')
#
#     def step_2_of_component(self, n):
#         if n == 0:
#             self.step_2_of_component_0()
#         else:
#             self.step_2_of_component_n(n)
#
#     def step_2_of_component_0(self):
#         """ Step 2 of component 0: find optimal horizontal line.
#
#         The horizontal line is parametrized as mean_0 + t * vec_0, where vec_0 is a unit
#         horizontal vector at mean_0. This step minimizes the cost function wrt mean_0 and
#         vec_0 under both constraints (vec_0 is unit norm and horizontal) using the default
#         SLSQP (Sequential Least Squares Programming) method of scipy minimize.
#         """
#         dim = self.dim
#
#         def func2min(x):
#             vec = x[:dim ** 2].reshape((dim, dim)).T
#             mean = x[dim ** 2:].reshape((dim, dim)).T
#             return cost_func(self.points, mean, vec)
#
#         def horizontality_constraint(x):
#             mean = x[dim ** 2:].reshape((dim, dim)).T
#             mat_cons = []
#             for i in range(dim - 1):
#                 for j in range(i + 1, dim):
#                     row = np.zeros(dim ** 2)
#                     row[i * dim: (i + 1) * dim] = mean[:, j]
#                     row[j * dim: (j + 1) * dim] = - mean[:, i]
#                     mat_cons.append(row)
#             mat_cons = np.vstack(mat_cons)
#             return mat_cons @ x[:dim ** 2]
#
#         cons = (
#             {'type': 'eq', 'fun': lambda x: np.sum(x[:dim ** 2] ** 2) - 1},
#             {'type': 'eq', 'fun': horizontality_constraint},
#         )
#         x0 = np.hstack((self.vec[0].T.reshape(dim * dim), self.mean[0].T.reshape(dim * dim)))
#         h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons) #, jac=jac) #, options={'disp': True}) #,method='trust-constr'
#         x_sol = h['x']
#         if not h['success']:
#             print('warning: failure in step 2 of component 0')
#         self.vec[0] = x_sol[:dim ** 2].reshape((dim, dim)).T
#         self.mean[0] = x_sol[dim ** 2:].reshape((dim, dim)).T
#
#     def step_2_of_component_1(self):
#         """ Step 2 of component 1: find optimal horizontal line.
#
#         The horizontal lift of the second geodesic component should intersect a lift of the
#         first geodesic component, and be orthogonal to it. For simplicity, we impose that these
#         two lifted components (i.e. horizontal straight lines in GL(2)) intersect. Therefore,
#         the lift of the second component is paramatrized by t -> mean_1 + t * vec_1, where
#         mean_1 = mean_0 + time * vec_0, under the constraints:
#         (1) vec_1 is a horizontal vector at mean_1
#         (2) vec_1 has unit norm
#         (3) vec_1 is orthogonal to vec_0.
#         This step minimizes the cost function over variables time and vec_1 under these 3 constraints,
#         using the default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
#         """
#         def func2min(x):
#             vec = x[:self.dim ** 2].reshape((self.dim, self.dim)).T
#             time = x[-1]
#             mean = self.mean[0] + time * self.vec[0]
#             return cost_func(self.points, mean, vec)
#
#         def horizontality_constraint(x):
#             mean = self.mean[0] + x[-1] * self.vec[0]
#             mat_cons = []
#             for i in range(self.dim - 1):
#                 for j in range(i + 1, self.dim):
#                     row = np.zeros(self.dim ** 2)
#                     row[i * self.dim: (i + 1) * self.dim] = mean[:, j]
#                     row[j * self.dim: (j + 1) * self.dim] = - mean[:, i]
#                     mat_cons.append(row)
#             mat_cons = np.vstack(mat_cons)
#             return mat_cons @ x[:self.dim ** 2]
#
#         cons = [
#             {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] ** 2) - 1},
#             {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] * self.vec[0].T.reshape(self.dim ** 2))},
#             {'type': 'eq', 'fun': horizontality_constraint},
#         ]
#         x0 = np.hstack([self.vec[1].T.reshape(self.dim ** 2), [self.time]])
#         h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
#         x_sol = h['x']
#         if not h['success']:
#             print('warning: failure in step 2 of component 1')
#         self.vec[1] = x_sol[:self.dim ** 2].reshape((self.dim, self.dim)).T
#         self.time = x_sol[-1]
#         self.mean[1] = self.mean[0] + self.time * self.vec[0]
#
#     def step_2_of_component_n(self, n):
#         """ Step 2 of component n (n>=2): find optimal horizontal line.
#
#         The following geodesic components are parametrized by t -> mean_n + t * vec_n, where
#         mean_n = mean_1 and
#         (1) vec_n is a horizontal vector at mean_n
#         (2) vec_n has unit norm
#         (3...) vec_n is orthogonal to vec_{n-1}, ..., vec_0.
#         This step minimizes the cost function over vec_n under these constraints, using the
#         default SLQSP (Sequential Least Squares Programming) method of scipy minimize.
#         """
#         def func2min(x):
#             vec = x.reshape((self.dim, self.dim)).T
#             return cost_func(self.points, self.mean[n], vec)
#
#         def horizontality_constraint(x):
#             mat_cons = []
#             for i in range(self.dim - 1):
#                 for j in range(i + 1, self.dim):
#                     row = np.zeros(self.dim ** 2)
#                     row[i * self.dim: (i + 1) * self.dim] = self.mean[n][:, j]
#                     row[j * self.dim: (j + 1) * self.dim] = - self.mean[n][:, i]
#                     mat_cons.append(row)
#             mat_cons = np.vstack(mat_cons)
#             return mat_cons @ x
#
#         def orthogonality_constraints(x):
#             vecs = np.stack([self.vec[k].T.reshape(self.dim ** 2) for k in range(n)])
#             return vecs @ x
#
#         cons = [
#             {'type': 'eq', 'fun': lambda x: np.sum(x[:self.dim ** 2] ** 2) - 1},
#             {'type': 'eq', 'fun': horizontality_constraint},
#             {'type': 'eq', 'fun': orthogonality_constraints},
#         ]
#         x0 = self.vec[n].T.reshape(self.dim ** 2)
#         h = scipy.optimize.minimize(func2min, x0=x0, constraints=cons)
#         x_sol = h['x']
#         if not h['success']:
#             print('warning: failure in step 2 of component', n)
#         self.vec[n] = x_sol.reshape((self.dim, self.dim)).T
#
#     def fit(self, points_spd):
#         """ Perform Bures-Wasserstein Principal Geodesic Analysis on SPD matrices.
#         """
#         self.points_spd = points_spd
#         self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
#         for n in range(self.n_comp):
#             self.component(n)
#
#         self.mean_spd = project(self.mean[1])
#         self.vecs_spd = np.stack([tangent_project(self.vec[n], self.mean[1]) for n in range(self.n_comp)])
#         self.costs, self.variances_spd = evaluate_results(self.components, self.points_spd, self.mean_spd)
#
#         indices = np.argsort(self.costs)
#         self.vecs_spd = self.vecs_spd[indices]
#         self.components = self.components[indices]
#         self.costs = self.costs[indices]
#         self.variances_spd = self.variances_spd[indices]
#
#         return self
