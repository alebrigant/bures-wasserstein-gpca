from tools.compute import *
from geomstats.geometry.spd_matrices import SPDLogEuclideanMetric


TIMES = np.linspace(0., 1., 20)


def generate_spd_matrices_on_orthogonal_geodesics(dim, n_points=21, n_geod=None, seed=None):
    if n_geod is None:
        n_geod = dim * (dim + 1) // 2
    if dim == 2:
        ratios = np.array([0.7, 0.6, 0.5])[:n_geod]
    elif dim == 3:
        ratios = np.arange(1., 0.2, -0.15)[:n_geod]
    else:
        print('Warning: not implemented for dim > 3')
    space_spd = SPDMatrices(dim)
    space_spd.equip_with_metric(SPDBuresWassersteinMetric)
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    mean = space_spd.random_point()
    vecs = []
    time_max = np.zeros(n_geod)
    for i in range(n_geod):
        vec_i = space_spd.random_tangent_vec(mean)
        for vec in vecs:
            vec_i = vec_i - space_spd.metric.inner_product(vec_i, vec, mean) * vec
        vec_i /= space_spd.metric.norm(vec_i, mean)
        vecs.append(vec_i)
        time_max[i] = compute_times_spd(vec_i, mean, n_points)[-1]
    indices = np.argsort(time_max)[::-1]
    time_max = time_max[indices] * ratios
    vecs = np.stack(vecs)[indices]

    points = []
    for i in range(n_geod):
        times = np.linspace(-time_max[i], time_max[i], n_points)
        points.append(space_spd.metric.geodesic(initial_point=mean, initial_tangent_vec=vecs[i])(times))
    return np.vstack(points), mean


def generate_spd_matrices_on_intersecting_geodesics(dim, n_times=20, ratio=1., eps=0.1, seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)

    if seed is not None: np.random.rand(seed)
    mean = spd_space.random_point()
    vec_1 = spd_space.random_tangent_vec(mean)
    vec_2 = spd_space.random_tangent_vec(mean)
    vec_1 /= spd_space.metric.norm(vec_1, mean)
    vec_2 /= spd_space.metric.norm(vec_2, mean)

    tmin_1, tmax_1 = compute_time_bounds_of_geodesic(mean, vec_1)
    tmin_2, tmax_2 = compute_time_bounds_of_geodesic(mean, vec_2)
    tmin = np.maximum(tmin_1, tmin_2)
    tmax = np.minimum(tmax_1, tmax_2)
    time = (1 - eps) * np.minimum(tmax, -tmin)

    times_1 = np.linspace(-time, time, n_times)
    times_2 = times_1 * ratio
    geod_1 = spd_space.metric.geodesic(initial_point=mean, initial_tangent_vec=vec_1)(times_1)
    geod_2 = spd_space.metric.geodesic(initial_point=mean, initial_tangent_vec=vec_2)(times_2)
    points_spd = np.vstack([geod_1, geod_2])

    return points_spd
