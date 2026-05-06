import numpy as np

from scipy.linalg import sqrtm, solve_lyapunov, solve_continuous_lyapunov, solve_sylvester, inv
from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric


def from_vec_to_sym(vec, dim):
    mat = np.zeros((dim, dim))
    for i in range(dim):
        mat[i, i:] = vec[i * dim - i * (i-1) // 2: (i+1) * dim - (i+1) * i // 2]
        mat[i, i] /= 2
    mat = mat + mat.T
    return mat


def from_sym_to_vec(mat, dim):
    vec = []
    for i in range(dim):
        vec.append(mat[i, i:])
    return np.hstack(vec)


def make_rotation_2d(angle):
    # Returns 2d matrix rotation from angle. Vectorized.
    if np.array(angle).ndim == 1:
        return np.stack([
            np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            for a in np.array(angle)
        ])
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def points_from_angles_2d(angles, sq_roots):
    rotations = make_rotation_2d(angles)
    return sq_roots @ rotations


def project(mat):
    # Project matrix of GL(n) into SPD(n). Vectorized.
    if mat.ndim == 3:
        return np.stack([m @ m.T for m in mat])
    return mat @ mat.T


def tangent_project(vec, mat):
    # Differential of projection
    if vec.ndim == 3:
        return np.stack([v @ mat.T + mat @ v.T for v in vec])
    return vec @ mat.T + mat @ vec.T


def monge_map_unit(cov_a, cov_b):
    # Compute the only symmetric positive definite
    # matrix T such that T @ cov_a @ T^t = cov_b.
    return np.linalg.inv(cov_a) @ sqrtm(cov_a @ cov_b)


def monge_map(cov_a, cov_b):
    # Find the element of the fiber over cov (matrix in GL(n))
    # that is closest to ref_mat. Vectorized.
    if cov_a.ndim == 3 and cov_b.ndim == 3:
        return np.stack([monge_map_unit(m_a, m_b) for (m_a, m_b) in zip(cov_a, cov_b)])
    if cov_a.ndim == 3 and cov_b.ndim == 2:
        return np.stack([monge_map_unit(m_a, cov_b) for m_a in cov_a])
    if cov_a.ndim == 2 and cov_b.ndim == 3:
        return np.stack([monge_map_unit(cov_a, m_b) for m_b in cov_b])
    return monge_map_unit(cov_a, cov_b)


def compute_time_bounds_of_geodesic(point, vec):
    ''' dim 2 !!'''
    vec_0 = solve_continuous_lyapunov(point, vec)
    #lbd_min, lbd_max = np.linalg.eigh(vec_0)[0]
    eigval, _ = np.linalg.eigh(vec_0)
    lbd_min, lbd_max = eigval[0], eigval[-1]
    t_min = (- 1 / lbd_max) * (lbd_max > 0) - 1e6 * (lbd_max <= 0)
    t_max = (- 1 / lbd_min) * (lbd_min < 0) + 1e6 * (lbd_min >= 0)
    return np.array([t_min, t_max])


def compute_times_spd(vec, base_point, n_points):
    ''' dim 2 !!'''
    sym_mat = solve_sylvester(base_point, base_point, vec)
    eigval, _ = np.linalg.eigh(sym_mat)
    lbd_min, lbd_max = eigval[0], eigval[-1]
    time_inf = - 1 / lbd_max if lbd_max > 0 else -np.infty
    time_sup = - 1 / lbd_min if lbd_min < 0 else np.infty
    time_bound = 0.8 * np.minimum(- time_inf, time_sup)
    return np.linspace(-time_bound, time_bound, n_points)


def compute_geodesic(point, vec, n_times=100, t_min_clip=-1e6, t_max_clip=1e6):
    ''' dim 2 !!'''
    t_min, t_max = compute_time_bounds_of_geodesic(point, vec)
    t_min = np.maximum(t_min, t_min_clip)
    t_max = np.minimum(t_max, t_max_clip)
    times = np.linspace(t_min, t_max, n_times)
    spd_space = SPDMatrices(2)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    return spd_space.metric.geodesic(initial_point=point, initial_tangent_vec=vec)(times)


def compute_costs_and_variances(components, points_spd, mean_spd):
    dim = points_spd.shape[-1]
    space = SPDMatrices(dim)
    space.equip_with_metric(SPDBuresWassersteinMetric)
    costs = np.array([np.sum(space.metric.dist(points_spd, component) ** 2) for component in components])
    variances = np.array([np.sum(space.metric.dist(component, mean_spd) ** 2) for component in components])
    return costs, variances


