import numpy as np

from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric
from numpy.linalg import norm
from ot.gaussian import bures_wasserstein_barycenter
from scipy.linalg import sqrtm
from tools.compute import project, align


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

        spd_space = SPDMatrices(dim)
        spd_space.equip_with_metric(SPDBuresWassersteinMetric)
        cost_1 = np.sum(spd_space.metric.dist(points_spd, components[0]) ** 2)
        cost_2 = np.sum(spd_space.metric.dist(points_spd, components[1]) ** 2)

        var_1 = np.sum(spd_space.metric.squared_dist(mean_spd, components[0])) / n_points
        var_2 = np.sum(spd_space.metric.squared_dist(mean_spd, components[1])) / n_points

        res = {'components': components, 'mean_spd': mean_spd,
               'vecs_spd': vecs_spd, 'cost_1': cost_1, 'cost_2': cost_2,
               'points': points, 'mean': mean, 'var_1': var_1, 'var_2': var_2}
        return res
