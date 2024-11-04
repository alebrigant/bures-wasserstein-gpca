import numpy as np

from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric
from scipy.linalg import sqrtm
from tools.compute import bures_wasserstein_sectional_curvature


TIMES = np.linspace(0., 1., 20)


def generate_spd_matrices_on_geodesic(dim, times=TIMES, seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    spd_1, spd_2 = spd_space.random_point(2)
    points_spd = spd_space.metric.geodesic(initial_point=spd_1, end_point=spd_2)(times)
    return points_spd


def generate_spd_matrices_on_orthogonal_geodesics(dim, times_1=TIMES, times_2=TIMES, seed=None):
    space_spd = SPDMatrices(dim)
    space_spd.equip_with_metric(SPDBuresWassersteinMetric)
    metric = space_spd.metric
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    mean = space_spd.random_point()
    vec_1 = space_spd.random_tangent_vec(mean)
    vec_1 /= metric.norm(vec_1, mean)
    vec_2 = space_spd.random_tangent_vec(mean)
    vec_2 = vec_2 - metric.inner_product(vec_1, vec_2, mean) * vec_1
    vec_2 /= metric.norm(vec_2, mean)
    vecs_1 = np.stack([t * vec_1 for t in times_1])
    vecs_2 = np.stack([t * vec_2 for t in times_2])
    geodesic_1 = metric.exp(vecs_1, mean)
    geodesic_2 = metric.exp(vecs_2, mean)
    return np.vstack((geodesic_1, geodesic_2)), mean


def generate_spd_matrices_on_intersecting_geodesics(dim, n_times=20, time=0.5, ratio=1., seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)

    if seed is not None: np.random.rand(seed)
    mean = spd_space.random_point()
    vec_1 = spd_space.random_tangent_vec(mean)
    vec_2 = spd_space.random_tangent_vec(mean)
    vec_1 /= spd_space.metric.norm(vec_1, mean)
    vec_2 /= spd_space.metric.norm(vec_2, mean)
    curvature = bures_wasserstein_sectional_curvature(vec_1, vec_2, mean)

    times_1 = np.linspace(-time, time, n_times)
    times_2 = times_1 * ratio
    geod_1 = spd_space.metric.geodesic(initial_point=mean, initial_tangent_vec=vec_1)(times_1)
    geod_2 = spd_space.metric.geodesic(initial_point=mean, initial_tangent_vec=vec_2)(times_2)
    points_spd = np.vstack([geod_1, geod_2])

    return points_spd, mean, curvature


def generate_initialization(points_spd):
    n_points = points_spd.shape[0]
    dim = points_spd.shape[-1]
    sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
    rotations = np.tile(np.eye(dim), (n_points, 1, 1))
    points = sq_roots @ rotations
    mean_init = np.sum(points, axis=0) / n_points
    vec_aux = np.random.rand(dim, dim) # np.eye(2) * 0.5
    sym_init = vec_aux + vec_aux.T
    vec_init = sym_init @ mean_init
    vec_init /= np.linalg.norm(vec_init)
    return mean_init, vec_init
