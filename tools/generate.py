import numpy as np

from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric
from scipy.linalg import sqrtm
from tools.compute import bures_wasserstein_sectional_curvature, bures_wasserstein_ricci_curvature


TIMES = np.linspace(0., 1., 20)


def generate_spd_matrices_on_geodesic(dim, times=TIMES, seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    spd_1, spd_2 = spd_space.random_point(2)
    points_spd = spd_space.metric.geodesic(initial_point=spd_1, end_point=spd_2)(times)
    return points_spd


def generate_spd_matrices_on_two_orthogonal_geodesics(dim, times_1=TIMES, times_2=TIMES, seed=None):
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


def generate_spd_matrices_on_orthogonal_geodesics(dim, times=None, seed=None):
    rdim = dim * (dim + 1) // 2
    if times is None:
        times = np.tile(TIMES, (rdim, 1))
    space_spd = SPDMatrices(dim)
    space_spd.equip_with_metric(SPDBuresWassersteinMetric)
    metric = space_spd.metric
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    mean = space_spd.random_point()
    vecs = []
    geodesics = []
    for i in range(rdim):
        vec_i = space_spd.random_tangent_vec(mean)
        for vec in vecs:
            vec_i = vec_i - metric.inner_product(vec_i, vec, mean) * vec
        vec_i /= metric.norm(vec_i, mean)
        vecs_i = np.stack([t * vec_i for t in times[i]])
        geod_i = metric.exp(vecs_i, mean)
        vecs.append(vec_i)
        geodesics.append(geod_i)
    return np.vstack(geodesics), mean


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

    velocity_1 = 2 * time * (n_times - 1) * spd_space.metric.log(geod_1[1:], geod_1[:-1])
    velocity_2 = 2 * time * (n_times - 1) * spd_space.metric.log(geod_2[1:], geod_2[:-1])
    ricci_1 = bures_wasserstein_ricci_curvature(velocity_1, geod_1[:-1])
    ricci_2 = bures_wasserstein_ricci_curvature(velocity_2, geod_2[:-1])
    ricci = np.hstack((ricci_1, ricci_2))
    return points_spd, mean, curvature, ricci


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
