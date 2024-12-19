import numpy as np
import scipy

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric
from numpy.linalg import norm
from ot.gaussian import bures_wasserstein_barycenter
from scipy.linalg import sqrtm
from tools.compute import align, make_rotation_2d, points_from_angles_2d, project, tangent_project


def project_on_line(points, mean, vec):
    times = np.stack([np.sum((pt - mean) * vec) for pt in points])
    eigval, _ = np.linalg.eigh(vec @ np.linalg.inv(mean))
    lbd_min, lbd_max = eigval
    time_inf = - 1 / lbd_max if lbd_max > 0 else -np.infty
    time_sup = - 1 / lbd_min if lbd_min < 0 else np.infty
    times = np.minimum(times, time_sup)
    times = np.maximum(times, time_inf)
    return np.stack([mean + t * vec for t in times])


def _cost_func(points, mean, vec):
    # Sum of squared norms of the residuals of the projections of points
    # on the line going through mean, directed by vec.
    projections = project_on_line(points, mean, vec)
    residuals = points - projections
    sq_norms = np.stack([np.sum(res ** 2) for res in residuals])
    return np.sum(sq_norms)


def compute_component(points, mean, vec):
    projections = project_on_line(points, mean, vec)
    return project(projections)


def evaluate_results(components, points_spd, mean_spd):
    dim = points_spd.shape[-1]
    space = SPDMatrices(dim)
    space.equip_with_metric(SPDBuresWassersteinMetric)

    costs = [np.sum(space.metric.dist(points_spd, component) ** 2) for component in components]
    variances = [np.sum(space.metric.dist(component, mean_spd) ** 2) for component in components]
    return np.array(costs), np.array(variances)


def check_boundary_2d(points, mean, vec, tol=1e-3):
    times = np.stack([np.sum((pt - mean) * vec) for pt in points])
    sym_mat = vec @ np.linalg.inv(mean)
    eigval, _ = np.linalg.eigh(sym_mat)
    lbd_min, lbd_max = eigval
    time_inf = - 1 / lbd_max if lbd_max > 0 else -np.infty
    time_sup = - 1 / lbd_min if lbd_min < 0 else np.infty
    bool = ((times - time_inf < - tol) + (time_sup - times < - tol))
    return np.sum(bool)


def postprocess_2d(costs, variances, components, boundary_checks):
    status = 1
    message = 'success'
    indices = np.argsort(costs)
    if np.any(indices != np.arange(3)):
        status = 2
        message = 'warning: components were reordered'
        costs, variances, components = costs[indices], variances[indices], components[indices]
    if np.any(boundary_checks):
        status = 0
        message = 'fail: at least one component line leaves the space of invertible matrices'

    return costs, variances, components, status, message


class BuresWassersteinTPCA:
    def __init__(self):
        pass

    @staticmethod
    def fit(points_spd):
        n_points, dim = points_spd.shape[:2]
        _, mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)
        mean = sqrtm(mean_spd)
        points = align(points_spd, mean)

        logs = (points - mean).reshape((n_points, dim ** 2))
        test_center = norm(np.sum(logs, axis=0) / n_points)
        if test_center > 1e-5:
            print(f'Warning: the norm of the mean of the tangent vectors is {test_center}, not zero.')
        covariance_of_logs = 1 / n_points * logs.T @ logs
        _, eig_vecs = np.linalg.eigh(covariance_of_logs)

        vecs_spd = np.zeros((dim ** 2, dim, dim))
        components = np.zeros((dim ** 2, n_points, dim, dim))
        for i in range(dim ** 2):
            vecs_spd[i] = eig_vecs[:, -i-1].reshape((dim, dim))
            scalar_prods_1 = np.stack([np.sum((pt - mean) * vecs_spd[i]) for pt in points])
            projections_on_line = np.stack([mean + t * vecs_spd[i] for t in scalar_prods_1])
            components[i] = project(projections_on_line)

        costs, variances = evaluate_results(components, points_spd, mean_spd)

        res = {'costs': costs, 'variances': variances, 'components': components,
               'mean_spd': mean_spd, 'vecs_spd': vecs_spd}
        return res


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
        verbose=False,
    ):
        """ Principal geodesic analysis for the Bures-Wasserstein metric in dimension 2.
        """
        self.max_iter = max_iter
        self.tol_1 = tol_1
        self.tol_2 = tol_2,
        self.verbose = verbose
        self.angles = None
        self.points = None
        self.sq_roots = None
        self.mean = None
        self.vec = None
        self.time = None
        self.fiber_angle = None

    def cost_func(self, n):
        """Cost function of component n_component.
        """
        return _cost_func(self.points, self.mean[n], self.vec[n])

    def component(self, n):
        """ First geodesic component.
        """
        self.component_init(n)
        costs = [self.cost_func(n)]

        for iteration in range(self.max_iter):
            self.component_step_1(n)
            self.component_step_2(n)

            costs.append(self.cost_func(n))
            if np.abs(costs[-1] - costs[-2]) < self.tol_1 or np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol_1:
                break

        if iteration == self.max_iter - 1:
            print(f'Warning: max number of iterations reached for component {n+1}.')

        component = compute_component(self.points, self.mean[n], self.vec[n])
        boundary_check = check_boundary_2d(self.points, self.mean[n], self.vec[n])
        return component, boundary_check, costs[-1]

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
            self.angles = np.zeros(n_points)
            self.points = points_from_angles_2d(self.angles, self.sq_roots)

            mean_1 = np.sum(self.points, axis=0) / n_points
            self.mean = np.stack([mean_1, np.zeros((2, 2)), np.zeros((2, 2))])

            vec_aux = np.random.rand(2, 2)
            vec_1 = (vec_aux + vec_aux.T) @ mean_1
            vec_1 /= norm(vec_1)
            self.vec = np.stack([vec_1, np.zeros((2, 2)), np.zeros((2, 2))])
        elif n == 1:
            self.time = 0.
            self.fiber_angle = np.zeros(2)
            self.mean[1] = self.mean[0].copy()

            vec_aux = np.random.rand(2, 2)
            vec_2 = (vec_aux + vec_aux.T) @ self.mean[1]
            vec_2 = vec_2 - np.sum(vec_2 * self.vec[0]) * self.vec[0]
            self.vec[1] = vec_2 / norm(vec_2)
        else:
            self.mean[2] = self.mean[1].copy()
            vec_aux = np.random.rand(2, 2)
            vec_3 = (vec_aux + vec_aux.T) @ self.mean[2]
            vec_1_mean_3 = self.vec[0] @ make_rotation_2d(self.fiber_angle[0])
            vec_2_mean_3 = self.vec[1]
            vec_3 = vec_3 - np.sum(vec_3 * vec_1_mean_3) * vec_1_mean_3 - np.sum(vec_3 * vec_2_mean_3) * vec_2_mean_3
            self.vec[2] = vec_3 / norm(vec_3)

    def component_step_1(self, n):
        """ Step 1 of component k: find optimal fiber representatives.

        The optimal representatives are parametrized as rotations(angles) @ sq_roots.
        The optimal angles are found by minimizing the cost function with the default
        BFGS method of scipy minimize.
        """
        def func2min(x):
            points = points_from_angles_2d(x, self.sq_roots)
            return _cost_func(points, self.mean[n], self.vec[n])

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
            return _cost_func(self.points, mean, vec)

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
            return _cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            a = self.mean[0].reshape(4)
            v = self.vec[0].reshape(4)
            return (
                    x[0] * (- (a[0] + x[-1] * v[0]) * np.sin(x[-2]) + (a[1] + x[-1] * v[1]) * np.cos(x[-2])) +
                    x[2] * (- (a[2] + x[-1] * v[2]) * np.sin(x[-2]) + (a[3] + x[-1] * v[3]) * np.cos(x[-2])) -
                    x[1] * ((a[0] + x[-1] * v[0]) * np.cos(x[-2]) + (a[1] + x[-1] * v[1]) * np.sin(x[-2])) -
                    x[3] * ((a[2] + x[-1] * v[2]) * np.cos(x[-2]) + (a[3] + x[-1] * v[3]) * np.sin(x[-2]))
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
            return _cost_func(self.points, mean, vec)

        def horizontality_constraint(x):
            a = self.mean[1].reshape(4)
            return (
                    x[0] * (- a[0] * np.sin(x[-2]) + a[1] * np.cos(x[-2])) +
                    x[2] * (- a[2] * np.sin(x[-2]) + a[3] * np.cos(x[-2])) -
                    x[1] * (a[0] * np.cos(x[-2]) + a[1] * np.sin(x[-2])) -
                    x[3] * (a[2] * np.cos(x[-2]) + a[3] * np.sin(x[-2]))
            )

        def orthogonality_constraints(x):
            rotation_0 = make_rotation_2d(self.fiber_angle[0])
            rotation_1 = make_rotation_2d(x[4])
            vec_0_rotated = (self.vec[0] @ rotation_0 @ rotation_1).reshape(4)
            vec_1_rotated = (self.vec[1] @ rotation_0).reshape(4)
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
        self.sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        n_points = self.sq_roots.shape[0]

        components = np.zeros((3, n_points, 2, 2))
        boundary_checks = np.zeros(3)
        costs_gl = np.zeros(3)
        for n in range(3):
            components[n], boundary_checks[n], costs_gl[n] = self.component(n)

        mean_spd = project(self.mean[1])
        vecs_spd = np.stack([
            tangent_project(self.vec[0] @ make_rotation_2d(self.fiber_angle[0]), self.mean[1]),
            tangent_project(self.vec[1], self.mean[1]),
            tangent_project(self.vec[2], self.mean[2])
        ])
        costs, variances = evaluate_results(components, points_spd, mean_spd)
        costs, variances, components, status, message = postprocess_2d(costs, variances, components, boundary_checks)

        return {'costs': costs, 'variances': variances, 'components': components,
                'mean_spd': mean_spd, 'vecs_spd': vecs_spd, 'status': status, 'message': message,
                'boundary_checks': boundary_checks, 'costs_gl': costs_gl}


class BuresWassersteinPGAND:
    def __init__(
        self,
        dim,
        max_iter=100,
        max_iter_gd=1000,
        tol=1e-3,
        tol_1=1e-3,
        tol_2=1e-3,
        tol_gd=1e-5,
        step_size_1=0.01,
        step_size_2=0.005,
        verbose=False,
        super_verbose=False,
        points_init=None,
        mean_init=None,
        vec_1_init=None,
        vec_2_init=None,
    ):
        """Principal geodesic analysis for the Bures-Wasserstein metric.
        """
        self.max_iter=max_iter
        self.max_iter_gd=max_iter_gd
        self.tol=tol
        self.tol_1=tol_1
        self.tol_2=tol_2
        self.tol_gd=tol_gd
        self.step_size_1=step_size_1
        self.step_size_2=step_size_2
        self.verbose=verbose
        self.super_verbose=super_verbose
        self.points_init=points_init
        self.mean_init=mean_init
        self.vec_1_init=vec_1_init
        self.vec_2_init=vec_2_init
        self.space = SPDMatrices(dim)
        self.space.equip_with_metric(SPDBuresWassersteinMetric)

    def component_1(self, rotations, sq_roots):
        """ First geodesic component.
        """
        if self.verbose: print('--------------- Component 1')

        if self.points_init is None:
            # initialize fiber representatives
            self.points_init = sq_roots @ rotations

        if self.mean_init is None:
            # initialize mean to be their Euclidean mean
            n_points = rotations.shape[0]
            self.mean_init = np.sum(self.points_init, axis=0) / n_points

        if self.vec_1_init is None:
            # randomly initialize horizontal vec at mean
            dim = rotations.shape[-1]
            vec_aux = np.random.rand(dim, dim)
            self.vec_1_init = (vec_aux + vec_aux.T) @ self.mean_init
            self.vec_1_init /= norm(self.vec_1_init)

        points = self.points_init.copy()
        mean_1 = self.mean_init.copy()
        vec_1 = self.vec_1_init.copy()

        costs = [_cost_func(points, mean_1, vec_1)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)

            # Compute new points in the fibers: align with the horizontal line
            rotations = self.component_k_step_1(rotations, sq_roots, mean_1, vec_1)
            points = sq_roots @ rotations
            if self.verbose: print(f'end of step 1, cost is {_cost_func(points, mean_1, vec_1)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            mean_1, vec_1 = self.component_1_step_2(points, mean_1, vec_1)
            costs.append(_cost_func(points, mean_1, vec_1))
            if self.verbose: print(f'end of step 2, cost is {costs[-1]}')

            if np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol_1 or np.abs(costs[-1] - costs[-2]) < self.tol_1:
                if self.verbose: print(f'Component 1: convergence reached in {iteration + 1} iterations.')
                break

        if iteration == self.max_iter - 1:
            print('Warning: max number of iterations reached for component 1.')

        scalar_prods = np.stack([np.sum((pt - mean_1) * vec_1) for pt in points])
        projections = np.stack([mean_1 + t * vec_1 for t in scalar_prods])
        component_1 = project(projections)

        return rotations, mean_1, vec_1, component_1, costs

    def component_k_step_1(self, rotations, sq_roots, mean, vec):
        """ Step 1 of component k: find optimal fiber representatives.

        The optimal representatives are parametrized as rotations @ sq_roots. The optimal
        rotations are found by Riemannian gradient descent on the cost function in the
        space of orthogonal matrices.
        """
        n_points = rotations.shape[0]
        new_rotations = []
        for i in range(n_points):
            new_rotation_i, cost_i = self.gradient_descent_component_k_step_1(rotations[i], sq_roots[i], mean, vec)
            new_rotations.append(new_rotation_i)
        return np.stack(new_rotations)

    def gradient_descent_component_k_step_1(self, rotation, sq_root, mean, vec):
        """ Gradient descent on the space of rotations to find optimal fiber representatives.
        """
        max_iter = self.max_iter_gd
        tol = self.tol_gd

        dim = rotation.shape[0]
        for iteration in range(max_iter):
            euclidean_grad = - 2 * sq_root.T @ (
                    mean + np.sum((sq_root @ rotation - mean) * vec) * vec - sq_root @ rotation
            )
            riemannian_grad = 1 / 2 * (euclidean_grad @ rotation.T - rotation @ euclidean_grad.T) @ rotation
            SOd = SpecialOrthogonal(dim)
            new_rotation = SOd.metric.exp(- self.step_size_1 * riemannian_grad, rotation)
            new_point = sq_root @ new_rotation
            projection = mean + np.sum((new_point - mean) * vec) * vec
            cost = np.sum((new_point - projection) ** 2)
            if norm(new_rotation - rotation) / norm(rotation) < tol or norm(new_rotation - rotation) < tol:
                return new_rotation, cost
            rotation = new_rotation.copy()

        if iteration == max_iter - 1:
            print(f'Warning (step 1): max number {max_iter} of iterations reached.')
        return rotation, cost

    @staticmethod
    def component_1_step_2(points, mean_1, vec_1):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal line is parametrized as mean_1 + t * vec_1, where vec_1 is a unit
        horizontal vector at mean_1. This step minimizes the cost function wrt mean_1 and
        vec_1 under both constraints (vec_1 is unit norm and horizontal) using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize.
        """
        dim = points.shape[1]

        def func2min(x, points):
            vec_1 = x[:dim ** 2].reshape((dim, dim)).T
            mean_1 = x[dim ** 2:].reshape((dim, dim)).T
            return _cost_func(points, mean_1, vec_1)

        def jac(x, points):
            vec_1 = x[:dim ** 2].reshape((dim, dim)).T
            mean_1 = x[dim ** 2:].reshape((dim, dim)).T
            sq_norm = np.sum(vec_1 ** 2)
            aux_1 = np.stack([mean_1 - pt + (sq_norm - 2) * np.sum((mean_1 - pt) * vec_1) * vec_1 for pt in points])
            aux_2 = np.stack([
                np.sum((mean_1 - pt) * vec_1) ** 2 * vec_1 + (sq_norm - 2) * np.sum((mean_1 - pt) * vec_1) * (mean_1 - pt)
                for pt in points
            ])
            grad_mean = 2 * np.sum(aux_1, axis=0).T.reshape(dim ** 2)
            grad_vec = 2 * np.sum(aux_2, axis=0).T.reshape(dim ** 2)
            return np.stack((grad_vec, grad_mean))

        def horizontality_constraint(x):
            mean_1 = x[dim ** 2:].reshape((dim, dim)).T
            mat_cons = []
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    row = np.zeros(dim ** 2)
                    row[i * dim: (i + 1) * dim] = mean_1[:, j]
                    row[j * dim: (j + 1) * dim] = - mean_1[:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x[:dim ** 2]

        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x[:dim ** 2] ** 2) - 1},
            {'type': 'eq', 'fun': horizontality_constraint},
        )

        x0 = np.hstack((vec_1.T.reshape(dim * dim), mean_1.T.reshape(dim * dim)))
        h = scipy.optimize.minimize(func2min, x0=x0, args=(points), constraints=cons, jac=jac) #, options={'disp': True}) #,method='trust-constr'
        x_sol = h['x']
        new_vec_1 = x_sol[:dim ** 2].reshape((dim, dim)).T
        new_mean_1 = x_sol[dim ** 2:].reshape((dim, dim)).T
        return new_mean_1, new_vec_1

    def component_2(self, rotations, sq_roots, mean_1, vec_1):
        """ Second geodesic component.
        """
        if self.verbose: print('--------------- Component 2')

        # initialize fiber representatives
        points = sq_roots @ rotations

        # Initialize the mean with value found in the first step
        dim = rotations.shape[-1]
        rotation_2 = np.eye(dim)
        time_2 = 0.
        mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2

        # Initialize unit horizontal vec_1-orthogonal vector
        if self.vec_2_init is None:
            vec_aux = np.random.rand(dim, dim)
            self.vec_2_init = (vec_aux + vec_aux.T) @ mean_1
        else:
            self.vec_2_init = self.vec_2_init @ np.linalg.inv(self.mean_init) @ mean_2

        vec_2 = self.vec_2_init - np.sum(self.vec_2_init * vec_1) * vec_1
        vec_2 /= norm(vec_2)

        costs = [_cost_func(points, mean_2, vec_2)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)

            # Compute new points in the fibers: align with the horizontal line
            rotations = self.component_k_step_1(rotations, sq_roots, mean_2, vec_2)
            points = sq_roots @ rotations
            if self.verbose: print(f'end of step 1, cost is {_cost_func(points, mean_2, vec_2)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            rotation_2, time_2, vec_2 = self.component_2_step_2(rotation_2, time_2, vec_2, points, mean_1, vec_1)
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            costs.append(_cost_func(points, mean_2, vec_2))
            if self.verbose: print(f'end of step 2, cost is {costs[-1]}')

            if np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol_2 or np.abs(costs[-1] - costs[-2]) < self.tol_2:
                if self.verbose: print(f'Component 2: convergence reached in {iteration + 1} iterations.')
                break

        if iteration == self.max_iter - 1:
            print('Warning: max number of iterations reached for component 1.')

        scalar_prods = np.stack([np.sum((pt - mean_2) * vec_2) for pt in points])
        projections = np.stack([mean_2 + t * vec_2 for t in scalar_prods])
        component_2 = project(projections)

        return rotations, mean_2, vec_2, rotation_2, component_2, costs

    def component_2_step_2(self, rotation_2, time_2, vec_2, points, mean_1, vec_1):
        """ Step 2 of component 2: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. Therefore it is parametrized by
        t -> mean_2 + t * vec_2, where mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2, and
        vec_2 is a unit horizontal vector at mean_2 and is orthogonal to vec_1 @ rotation_2.
        It is found by an alternate minimization of the cost function over rotation_2 (step
        2-1) and vec_2 and time_2 (step 2-2).
        """
        mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
        costs = [_cost_func(points, mean_2, vec_2)]
        for iteration in range(self.max_iter):
            if self.super_verbose: print(f'iteration {iteration} of step 2')
            new_rotation_2 = self.component_2_step_2_1(rotation_2, time_2, vec_2, points, mean_1, vec_1)

            vec_2, time_2 = self.component_2_step_2_2(new_rotation_2, rotation_2, time_2, vec_2, points, mean_1, vec_1)
            rotation_2 = new_rotation_2.copy()
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            costs.append(_cost_func(points, mean_2, vec_2))
            if self.super_verbose: print('step 2-2, cost is ', costs[-1])

            if np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol or np.abs(costs[-1] - costs[-2]) < self.tol:
                if self.verbose: print(f'Step 2: convergence reached in {iteration + 1} iterations.')
                break

        if iteration == self.max_iter - 1:
            print('Step 2: Warning! max number of iterations reached for step 2 of component 2.')
        return rotation_2, time_2, vec_2

    def component_2_step_2_1(self, rotation_2, time_2, vec_2, points, mean_1, vec_1):
        """ Step 2-1 of component 2: Riemannian gradient descent over rotation_2.
        """
        verbose = self.super_verbose
        max_iter = self.max_iter_gd
        step_size = self.step_size_2
        tol = self.tol

        dim = points.shape[-1]
        for iteration in range(max_iter):
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            if verbose: print('step 2-1, cost is ', _cost_func(points, mean_2, vec_2))
            scalar_prods = np.stack([np.sum((pt - mean_2) * vec_2) for pt in points])
            projections = np.stack([mean_2 + t * vec_2 for t in scalar_prods])
            sum_residuals = np.sum(projections - points, axis=0)
            grad = (mean_1 + time_2 * vec_1).T @ sum_residuals - \
                   rotation_2 @ sum_residuals.T @ (mean_1 + time_2 * vec_1) @ rotation_2

            SOd = SpecialOrthogonal(dim)
            new_rotation_2 = SOd.metric.exp(- step_size * grad, rotation_2)
            if norm(new_rotation_2 - rotation_2) / norm(rotation_2) < tol or norm(new_rotation_2 - rotation_2) < tol:
                return new_rotation_2
            rotation_2 = new_rotation_2.copy()

        if iteration == max_iter - 1:
            print('Warning (step 2-1): max number of iterations reached.')
        return rotation_2

    @staticmethod
    def component_2_step_2_2(rotation_2, old_rotation_2, time_2, vec_2, points, mean_1, vec_1):
        """ Step 2-2 of component 2: optimize over vec_2 and time_2.

        The horizontal line is parametrized by t -> mean_2 + t * vec_2 where
        mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2, and vec_2 is a unit horizontal
        vector at mean_2 orthogonal to vec_1 @ rotation_2. This step minimizes the cost
        function wrt vec_2 and time_2 under these three constraints using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize.
        """
        dim = points.shape[1]

        def func2min(x, rotation_2, points, mean_1, vec_1):
            vec_2 = x[:dim ** 2].reshape((dim, dim)).T
            time_2 = x[-1]
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            return _cost_func(points, mean_2, vec_2)

        def horizontality_constraint(x):
            mean_2 = (mean_1 + x[-1] * vec_1) @ rotation_2
            mat_cons = []
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    row = np.zeros(dim ** 2)
                    row[i * dim: (i + 1) * dim] = mean_2[:, j]
                    row[j * dim: (j + 1) * dim] = - mean_2[:, i]
                    mat_cons.append(row)
            mat_cons = np.vstack(mat_cons)
            return mat_cons @ x[:dim ** 2]

        vec_1_rotated = (vec_1 @ rotation_2).T.reshape(dim ** 2)
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:dim ** 2] ** 2) - 1},         # vec_2 must be unit norm
            {'type': 'eq', 'fun': lambda x: np.sum(x[:dim ** 2] * vec_1_rotated)},  # vec_2 must be orthogonal to vec_1
            {'type': 'eq', 'fun': horizontality_constraint},
        ]

        vec_2_init = vec_2 @ old_rotation_2.T @ rotation_2
        x0 = np.hstack([vec_2_init.T.reshape(dim ** 2), [time_2]])
        h = scipy.optimize.minimize(func2min, x0=x0, args=(rotation_2, points, mean_1, vec_1), constraints=cons) #, options={'disp': True})
        x_sol = h['x']
        new_vec_2 = x_sol[:dim ** 2].reshape((dim, dim)).T
        new_time_2 = x_sol[-1]
        return new_vec_2, new_time_2

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
    #     costs = [_cost_func(points, mean_3, vec_3)]
    #     for iteration in range(self.max_iter):
    #         if self.super_verbose: print(f'iteration {iteration} of step 2')
    #         rotation_3 = self.component_3_step_2_1(rotation_3, vec_3, points, mean_2, vec_1, vec_2)
    #
    #         vec_3 = self.component_2_step_2_2(rotation_2, rotation_2, time_2, vec_2, points, mean_1, vec_1)
    #         mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
    #         costs.append(_cost_func(points, mean_2, vec_2))
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
        n_points = points_spd.shape[0]
        dim = points_spd.shape[1]
        sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
        rotations = np.tile(np.eye(dim), (n_points, 1, 1))

        rotations, mean_1, vec_1, component_1, costs_1 = self.component_1(rotations, sq_roots)

        rotations, mean_2, vec_2, rotation_2, component_2, costs_2 = self.component_2(rotations, sq_roots, mean_1, vec_1)

        mean_spd = project(mean_2)

        spd_space = SPDMatrices(dim)
        spd_space.equip_with_metric(SPDBuresWassersteinMetric)
        cost_1 = np.sum(spd_space.metric.dist(points_spd, component_1) ** 2)
        cost_2 = np.sum(spd_space.metric.dist(points_spd, component_2) ** 2)
        vec_1_spd = tangent_project(vec_1 @ rotation_2, mean_2)
        vec_2_spd = tangent_project(vec_2, mean_2)

        costs = np.array([cost_1, cost_2])
        vecs_spd = np.stack([vec_1_spd, vec_2_spd])
        components = np.stack([component_1, component_2])

        var_1 = np.sum(spd_space.metric.squared_dist(mean_spd, component_1)) / n_points
        var_2 = np.sum(spd_space.metric.squared_dist(mean_spd, component_2)) / n_points

        return {
            'mean_spd': mean_spd, 'vecs_spd': vecs_spd,
            'costs': costs, 'components': components,
            'var_1': var_1, 'var_2': var_2,
        }
