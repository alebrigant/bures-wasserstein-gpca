from tools.compute import *


TIMES = np.linspace(0., 1., 20)


def generate_spd_matrices_on_geodesic(dim, times=TIMES, seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    spd_1, spd_2 = spd_space.random_point(2)
    points_spd = spd_space.metric.geodesic(initial_point=spd_1, end_point=spd_2)(times)
    return points_spd


# def generate_spd_matrices_on_two_orthogonal_geodesics(dim, times_1=TIMES, times_2=TIMES, seed=None):
#     space_spd = SPDMatrices(dim)
#     space_spd.equip_with_metric(SPDBuresWassersteinMetric)
#     metric = space_spd.metric
#     np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
#     mean = space_spd.random_point()
#     vec_1 = space_spd.random_tangent_vec(mean)
#     vec_1 /= metric.norm(vec_1, mean)
#     vec_2 = space_spd.random_tangent_vec(mean)
#     vec_2 = vec_2 - metric.inner_product(vec_1, vec_2, mean) * vec_1
#     vec_2 /= metric.norm(vec_2, mean)
#     vecs_1 = np.stack([t * vec_1 for t in times_1])
#     vecs_2 = np.stack([t * vec_2 for t in times_2])
#     geodesic_1 = metric.exp(vecs_1, mean)
#     geodesic_2 = metric.exp(vecs_2, mean)
#     return np.vstack((geodesic_1, geodesic_2)), mean


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

# def generate_spd_matrices_on_orthogonal_geodesics(dim, n_points, seed=None):
#     space_spd = SPDMatrices(dim)
#     space_spd.equip_with_metric(SPDBuresWassersteinMetric)
#     np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
#     rdim = dim * (dim + 1) // 2
#     mean = np.eye(dim) #np.random.randint(2, 5) * np.eye(dim)  #np.random.rand(dim, dim)
#     points_spd = []
#     vecs = []
#     time_bounds = np.zeros(rdim)
#     for n in range(rdim):
#         mat_aux = np.random.rand(dim, dim)
#         sym_mat = mat_aux + mat_aux.T
#         vec = sym_mat @ mean
#         if n > 0:
#             projections = np.stack([np.sum(vec * vecs[i]) * vecs[i] for i in range(n)])
#             vec = vec - np.sum(projections, axis=0)
#         vec /= np.linalg.norm(vec)
#         times = compute_times(vec, mean, n_points)
#         time_bounds[n] = times[-1]
#         vecs.append(vec)
#         line = mean + np.einsum('i,jk->ijk', times, vec)
#         if np.all(np.array([det(mean) * det(mat) for mat in line]) <= 0):
#             print('Passe par un trou !!')
#             print([det(mat) for mat in line])
#         geod = project(line)
#         points_spd.append(geod)
#
#     indices = np.argsort(time_bounds)[::-1]
#     points_spd = np.stack(points_spd)
#     points_spd = np.vstack(points_spd[indices])
#     return points_spd, project(mean)


# def generate_spd_matrices_on_orthogonal_geodesics(dim, times=None, seed=None):
#     rdim = dim * (dim + 1) // 2
#     if times is None:
#         times = np.tile(TIMES, (rdim, 1))
#     space_spd = SPDMatrices(dim)
#     space_spd.equip_with_metric(SPDBuresWassersteinMetric)
#     metric = space_spd.metric
#     np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
#     mean = space_spd.random_point()
#     vecs = []
#     geodesics = []
#     for i in range(rdim):
#         vec_i = space_spd.random_tangent_vec(mean)
#         for vec in vecs:
#             vec_i = vec_i - metric.inner_product(vec_i, vec, mean) * vec
#         vec_i /= metric.norm(vec_i, mean)
#         print(compute_times_spd(vec_i, mean, 21)[-1])
#         vecs_i = np.stack([t * vec_i for t in times[i]])
#         geod_i = metric.exp(vecs_i, mean)
#         vecs.append(vec_i)
#         geodesics.append(geod_i)
#     return np.vstack(geodesics), mean


# def generate_spd_matrices_on_orthogonal_geodesics(dim, times=None, seed=None):
#     rdim = dim * (dim + 1) // 2
#     if times is None:
#         times = np.tile(TIMES, (rdim, 1))
#     space_spd = SPDMatrices(dim)
#     space_spd.equip_with_metric(SPDBuresWassersteinMetric)
#     metric = space_spd.metric
#     np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
#     mean = space_spd.random_point()
#     vecs = []
#     geodesics = []
#     for i in range(rdim):
#         vec_i = space_spd.random_tangent_vec(mean)
#         for vec in vecs:
#             vec_i = vec_i - metric.inner_product(vec_i, vec, mean) * vec
#         vec_i /= metric.norm(vec_i, mean)
#         vecs_i = np.stack([t * vec_i for t in times[i]])
#         geod_i = metric.exp(vecs_i, mean)
#         vecs.append(vec_i)
#         geodesics.append(geod_i)
#     return np.vstack(geodesics), mean


def generate_spd_matrices_on_intersecting_geodesics(dim, n_times=20, ratio=1., eps=0.1, seed=None):
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)

    if seed is not None: np.random.rand(seed)
    mean = spd_space.random_point()
    vec_1 = spd_space.random_tangent_vec(mean)
    vec_2 = spd_space.random_tangent_vec(mean)
    vec_1 /= spd_space.metric.norm(vec_1, mean)
    vec_2 /= spd_space.metric.norm(vec_2, mean)
    curvature = bures_wasserstein_sectional_curvature(vec_1, vec_2, mean)

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
