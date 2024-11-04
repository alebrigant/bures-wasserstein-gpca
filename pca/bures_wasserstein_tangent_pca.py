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

        tangent_vecs = (points - mean).reshape((n_points, dim ** 2))
        test_center = norm(np.sum(tangent_vecs, axis=0) / n_points)
        if test_center > 1e-5:
            print(f'Warning: the norm of the mean of the tangent vectors is {test_center}, not zero.')
        cov_tangent_vecs = 1 / n_points * tangent_vecs.T @ tangent_vecs
        _, eig_vecs = np.linalg.eigh(cov_tangent_vecs)

        vec_1 = eig_vecs[:, -1].reshape((dim, dim))
        vec_2 = eig_vecs[:, -2].reshape((dim, dim))

        scalar_prods_1 = np.stack([np.sum((pt - mean) * vec_1) for pt in points])
        scalar_prods_2 = np.stack([np.sum((pt - mean) * vec_2) for pt in points])
        projections_1 = np.stack([mean + t * vec_1 for t in scalar_prods_1])
        projections_2 = np.stack([mean + t * vec_2 for t in scalar_prods_2])
        component_1 = project(projections_1)
        component_2 = project(projections_2)

        spd_space = SPDMatrices(dim)
        spd_space.equip_with_metric(SPDBuresWassersteinMetric)
        cost_1 = np.sum(spd_space.metric.dist(points_spd, component_1) ** 2)
        cost_2 = np.sum(spd_space.metric.dist(points_spd, component_2) ** 2)

        res = {'component_1': component_1, 'component_2': component_2, 'mean_spd': mean_spd,
               'vec_1': vec_1, 'vec_2': vec_2, 'cost_1': cost_1, 'cost_2': cost_2,
               'points': points, 'mean': mean}
        return res
