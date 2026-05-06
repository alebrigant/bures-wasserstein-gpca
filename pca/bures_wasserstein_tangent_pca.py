from ot.gaussian import bures_wasserstein_barycenter
from tools.compute import *


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
        n_points, dim = points_spd.shape[:2]
        self.points_spd = points_spd
        _, self.mean_spd = bures_wasserstein_barycenter(np.zeros((n_points, 2)), points_spd)

        points_sym = monge_map(self.mean_spd, points_spd) - np.tile(np.eye(dim), (n_points, 1, 1))
        components_sym, self.times, vecs_sym, _ = weighted_pca_on_symmetric_matrices(points_sym, self.mean_spd)

        mean = sqrtm(self.mean_spd)
        components_gl = mean + components_sym @ mean
        self.components = np.stack([project(comp) for comp in components_gl])
        self.vecs_spd = np.stack([tangent_project(vec @ mean, mean) for vec in vecs_sym])
        self.costs, self.variances = compute_costs_and_variances(self.components, points_spd, self.mean_spd)
        return self