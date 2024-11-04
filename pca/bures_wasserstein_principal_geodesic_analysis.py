import numpy as np
import scipy

from numpy.linalg import norm

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric
from tools.compute import make_rotation_2d, points_from_angles_2d, project, tangent_project


def cost_func(points, mean, vec):
    # Sum of squared norms of the residuals of the projections of points
    # on the line going through mean, directed by vec.
    scalar_prods = np.stack([np.sum((pt - mean) * vec) for pt in points])
    projections = np.stack([mean + t * vec for t in scalar_prods])
    residuals = points - projections
    sq_norms = np.stack([np.sum(res ** 2) for res in residuals])
    return np.sum(sq_norms)


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
            return BuresWassersteinPGAND(**kwargs)


class BuresWassersteinPGA2D:
    def __init__(
        self,
        max_iter=100,
        tol_1=1e-3,
        tol_2=1e-3,
        verbose=False,
        points_init=None,
        mean_init=None,
        vec_1_init=None,
        vec_2_init=None,
    ):
        """ Principal geodesic analysis for the Bures-Wasserstein metric in dimension 2.
        """
        self.max_iter=max_iter
        self.tol_1=tol_1
        self.tol_2=tol_2,
        self.verbose=verbose
        self.points_init = points_init
        self.mean_init = mean_init
        self.vec_1_init = vec_1_init
        self.vec_2_init = vec_2_init

    def component_1(self, angles, sq_roots):
        """ First geodesic component.
        """
        if self.verbose: print('--------------- component 1')
        if self.points_init is None:
            # initialize with square roots
            self.points_init = points_from_angles_2d(angles, sq_roots)

        if self.mean_init is None:
            # initialize mean to be Euclidean mean
            n_points = len(angles)
            self.mean_init = np.sum(self.points_init, axis=0) / n_points

        if self.vec_1_init is None:
            # randomly initialize horizontal vec at mean
            vec_aux = np.random.rand(2, 2)
            self.vec_1_init = (vec_aux + vec_aux.T) @ self.mean_init
            self.vec_1_init /= norm(self.vec_1_init)

        points = self.points_init.copy()
        mean = self.mean_init.copy()
        vec = self.vec_1_init.copy()

        costs = [cost_func(points, mean, vec)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)
            # Compute new points in the fibers: align with the horizontal line
            angles = self.component_k_step_1(angles, sq_roots, mean, vec)
            points = points_from_angles_2d(angles, sq_roots)
            if self.verbose: print(f'end of step 1, cost is {cost_func(points, mean, vec)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            mean, vec = self.component_1_step_2(points, mean, vec)
            costs.append(cost_func(points, mean, vec))
            if self.verbose: print(f'end of step 2, cost is {costs[-1]}')

            if np.abs(costs[-1] - costs[-2]) < self.tol_1 or np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol_1:
                if self.verbose: print(f'Component 1: convergence reached in {iteration + 1} iterations.')
                break

        if iteration == self.max_iter - 1:
            print('Warning: max number of iterations reached for component 1.')

        scalar_prods = np.stack([np.sum((pt - mean) * vec) for pt in points])
        projections = np.stack([mean + t * vec for t in scalar_prods])
        component_1 = project(projections)

        return angles, mean, vec, component_1, costs

    @staticmethod
    def component_k_step_1(angles, sq_roots, mean, vec):
        """ Step 1 of component k: find optimal fiber representatives.

        The optimal representatives are parametrized as rotations(angles) @ sq_roots.
        The optimal angles are found by minimizing the cost function with the default
        BFGS method of scipy minimize.
        """
        def func2min(x, sq_roots, mean, vec):
            points = points_from_angles_2d(x, sq_roots)
            return cost_func(points, mean, vec)

        h = scipy.optimize.minimize(func2min, x0=angles, args=(sq_roots, mean, vec))
        return h['x']

    @staticmethod
    def component_1_step_2(points, mean, vec):
        """ Step 2 of component 1: find optimal horizontal line.

        The horizontal line is parametrized as mean_1 + t * vec_1, where vec_1 is a unit
        horizontal vector at mean_1. This step minimizes the cost function wrt mean_1 and
        vec_1 under both constraints (vec_1 is unit norm and horizontal) using the default
        SLSQP (Sequential Least Squares Programming) method of scipy minimize. In the two-
        dimensional case, the horizontality condition reduces to one scalar equation.
        """
        def func2min(x, points):
            vec = x[:4].reshape((2, 2))
            mean = x[4:].reshape((2, 2))
            return cost_func(points, mean, vec)

        cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},  # unit norm
        {'type': 'eq', 'fun': lambda x: x[0] * x[5] - x[1] * x[4] + x[2] * x[7] - x[3] * x[6]}
    ]

        x0 = np.hstack([vec.reshape(4), mean.reshape(4)])
        h = scipy.optimize.minimize(func2min, x0=x0, args=(points), constraints=cons)
        x_sol = h['x']
        new_vec = x_sol[:4].reshape((2, 2))
        new_mean = x_sol[4:].reshape((2, 2))
        return new_mean, new_vec

    def component_2(self, angles, mean_1, vec_1, sq_roots):
        """ Second geodesic component.
        """
        if self.verbose: print('--------------- component 2')
        points = points_from_angles_2d(angles, sq_roots)

        # Initialize the mean with value found in the first step
        angle = 0.
        time = 0.
        mean_2 = (mean_1 + time * vec_1) @ make_rotation_2d(angle)

        # Initialize unit vector
        vec_2 = make_rotation_2d(np.pi / 2) @ vec_1

        costs = [cost_func(points, mean_2, vec_2)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)
            # Compute new points in the fibers: align with the line
            angles = self.component_k_step_1(angles, sq_roots, mean_2, vec_2)
            points = points_from_angles_2d(angles, sq_roots)
            if self.verbose: print(f'end of step 1, cost is {cost_func(points, mean_2, vec_2)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            vec_2, angle, time = self.component_2_step_2(points, mean_1, vec_1, vec_2, angle, time)
            mean_2 = (mean_1 + time * vec_1) @ make_rotation_2d(angle)
            costs.append(cost_func(points, mean_2, vec_2))
            if self.verbose: print(f'end of step 2, cost is {costs[-1]}')

            if np.abs(costs[-1] - costs[-2]) < self.tol_2 or np.abs((costs[-1] - costs[-2]) / costs[-2]) < self.tol_2:
                if self.verbose: print(f'Component 2: convergence reached in {iteration + 1} iterations.')
                break

        if iteration == self.max_iter - 1:
            print('Warning: max number of iterations reached for component 2.')

        scalar_prods = np.stack([np.sum((pt - mean_2) * vec_2) for pt in points])
        projections = np.stack([mean_2 + t * vec_2 for t in scalar_prods])
        component_2 = np.stack([proj @ proj.T for proj in projections])

        return mean_2, vec_2, angle, component_2, costs

    @staticmethod
    def component_2_step_2(points, mean, vec, vec_2, angle, time):
        """ Step 2 of component 2: find optimal horizontal line.

        The horizontal lift of the second geodesic component should intersect a lift of the
        first geodesic component, and be orthogonal to it. Therefore, it is parametrized by
        t -> mean_2 + t * vec_2, where mean_2 = (mean_1 + time * vec_1) @ rotation(angle),
        and vec_2 is a unit horizontal vector at mean_2 and is orthogonal to
        vec_1 @ rotation(angle). It is found by minimizing the cost function over angle, time,
        and vec_2 under these 3 constraints, using the default SLQSP (Sequential Least Squares
        Programming) method of scipy minimize.
        """
        def func2min(x, points, mean):
            vec_2 = x[:4].reshape((2, 2))
            angle = x[4]
            time = x[5]
            rotation = make_rotation_2d(np.squeeze(angle))
            mean_2 = (mean + time * vec) @ rotation
            return cost_func(points, mean_2, vec_2)

        a = mean.reshape(4)
        v = vec.reshape(4)
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x[:4] ** 2) - 1},
            {'type': 'eq', 'fun': lambda x: (
                    (v[0] * np.cos(x[-2]) + v[1] * np.sin(x[-2])) * x[0] +
                    (v[1] * np.cos(x[-2]) - v[0] * np.sin(x[-2])) * x[1] +
                    (v[2] * np.cos(x[-2]) + v[3] * np.sin(x[-2])) * x[2] +
                    (v[3] * np.cos(x[-2]) - v[2] * np.sin(x[-2])) * x[3]
            )},
            {'type': 'eq', 'fun': lambda x: (
                    x[0] * (- (a[0] + x[-1] * v[0]) * np.sin(x[-2]) + (a[1] + x[-1] * v[1]) * np.cos(x[-2])) +
                    x[2] * (- (a[2] + x[-1] * v[2]) * np.sin(x[-2]) + (a[3] + x[-1] * v[3]) * np.cos(x[-2])) -
                    x[1] * ((a[0] + x[-1] * v[0]) * np.cos(x[-2]) + (a[1] + x[-1] * v[1]) * np.sin(x[-2])) -
                    x[3] * ((a[2] + x[-1] * v[2]) * np.cos(x[-2]) + (a[3] + x[-1] * v[3]) * np.sin(x[-2]))
            )}
        )

        x0 = np.hstack([vec_2.reshape(4), angle, time])
        h = scipy.optimize.minimize(func2min, x0=x0, args=(points, mean), constraints=cons)
        x_sol = h['x']
        new_vec_2 = x_sol[:4].reshape((2, 2))
        new_angle = np.squeeze(x_sol[-2])
        new_rotation = make_rotation_2d(new_angle)
        new_time = np.squeeze(x_sol[-1])
        return new_vec_2, new_angle, new_time

    def fit(self, points_spd):
        n_points = points_spd.shape[0]
        sq_roots = np.stack([scipy.linalg.sqrtm(pt_spd) for pt_spd in points_spd])
        angles = np.zeros(n_points)

        angles, mean_1, vec_1, component_1, costs_1 = self.component_1(angles, sq_roots)

        mean_2, vec_2, angle, component_2, costs_2 = self.component_2(angles, mean_1, vec_1, sq_roots)

        mean_spd = project(mean_2)

        spd_space = SPDMatrices(2)
        spd_space.equip_with_metric(SPDBuresWassersteinMetric)
        cost_1 = np.sum(spd_space.metric.dist(points_spd, component_1)**2)
        cost_2 = np.sum(spd_space.metric.dist(points_spd, component_2)**2)
        vec_1_spd = tangent_project(vec_1 @ make_rotation_2d(angle), mean_2)
        vec_2_spd = tangent_project(vec_2, mean_2)

        return {
            'mean_spd': mean_spd, 'vec_1_spd': vec_1_spd, 'vec_2_spd': vec_2_spd, 'vec_1': vec_1, 'vec_2': vec_2,
            'costs_1': costs_1, 'costs_2': costs_2, 'cost_1': cost_1, 'cost_2': cost_2,
            'component_1': component_1, 'component_2': component_2,
        }


class BuresWassersteinPGAND:
    def __init__(
        self,
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

        costs = [cost_func(points, mean_1, vec_1)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)

            # Compute new points in the fibers: align with the horizontal line
            rotations = self.component_k_step_1(rotations, sq_roots, mean_1, vec_1)
            points = sq_roots @ rotations
            if self.verbose: print(f'end of step 1, cost is {cost_func(points, mean_1, vec_1)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            mean_1, vec_1 = self.component_1_step_2(points, mean_1, vec_1)
            costs.append(cost_func(points, mean_1, vec_1))
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
            return cost_func(points, mean_1, vec_1)

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

        costs = [cost_func(points, mean_2, vec_2)]
        if self.verbose: print(f'initial cost is {costs[-1]}')

        for iteration in range(self.max_iter):
            if self.verbose: print('-- iteration ', iteration)

            # Compute new points in the fibers: align with the horizontal line
            rotations = self.component_k_step_1(rotations, sq_roots, mean_2, vec_2)
            points = sq_roots @ rotations
            if self.verbose: print(f'end of step 1, cost is {cost_func(points, mean_2, vec_2)}')

            # Compute new horizontal line that maximizes the variance of the points' projections
            rotation_2, time_2, vec_2 = self.component_2_step_2(rotation_2, time_2, vec_2, points, mean_1, vec_1)
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            costs.append(cost_func(points, mean_2, vec_2))
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
        costs = [cost_func(points, mean_2, vec_2)]
        for iteration in range(self.max_iter):
            if self.super_verbose: print(f'iteration {iteration} of step 2')
            rotation_2 = self.component_2_step_2_1(rotation_2, time_2, vec_2, points, mean_1, vec_1)

            vec_2, time_2 = self.component_2_step_2_2(rotation_2, rotation_2, time_2, vec_2, points, mean_1, vec_1)
            mean_2 = (mean_1 + time_2 * vec_1) @ rotation_2
            costs.append(cost_func(points, mean_2, vec_2))
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
            if verbose: print('step 2-1, cost is ', cost_func(points, mean_2, vec_2))
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
            return cost_func(points, mean_2, vec_2)

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

    def fit(self, points_spd):
        """ Perform Bures-Wasserstein Principal Geodesic Analysis on SPD matrices.
        """
        n_points = points_spd.shape[0]
        dim = points_spd.shape[1]
        sq_roots = np.stack([scipy.linalg.sqrtm(pt_spd) for pt_spd in points_spd])
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

        return {
            'mean_spd': mean_spd, 'vec_1_spd': vec_1_spd, 'vec_2_spd': vec_2_spd, 'vec_1': vec_1, 'vec_2': vec_2,
            'costs_1': costs_1, 'costs_2': costs_2, 'cost_1': cost_1, 'cost_2': cost_2,
            'component_1': component_1, 'component_2': component_2,
        }
