from tools.compute import *
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


def test_make_rotation_2d_vectorization(tol=1e-5):
    angles = np.random.rand(5)
    result = make_rotation_2d(angles)
    expected = np.stack([make_rotation_2d(angle) for angle in angles])
    assert np.all(np.abs(result - expected) < tol)


def test_projection_vectorization(tol=1e-5):
    mats = np.random.rand(5, 2, 2)
    result = project(mats)
    expected = np.stack([project(mat) for mat in mats])
    assert np.all(np.abs(result - expected) < tol)


def test_align_vectorization(tol=1e-5):
    ref_mat = np.random.rand(2, 2)
    mats = np.random.rand(5, 2, 2)
    covs = project(mats)
    result = align(covs, ref_mat)
    expected = np.stack([align(cov, ref_mat) for cov in covs])
    assert np.all(np.abs(result - expected) < tol)


def test_align(tol=1e-5):
    spd_space = SPDMatrices(2)
    spd = spd_space.random_point()
    ref_mat = sqrtm(spd)
    spds = spd_space.random_point(5)
    mats = align(spds, ref_mat)
    assert np.all(np.abs(project(mats) - spds) < tol)


def test_bures_wasserstein_sectional_curvature(tol=1e-5):
    np.random.seed(1)
    dim = 2
    rot_space = SpecialOrthogonal(dim)
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    metric = spd_space.metric
    d = np.random.rand(dim)
    P = rot_space.random_point()
    base_point = P @ np.diag(d) @ P.T
    vec_1 = spd_space.random_tangent_vec(base_point)
    vec_2 = spd_space.random_tangent_vec(base_point)
    result = bures_wasserstein_sectional_curvature(vec_1, vec_2, base_point)

    vec_1 /= metric.norm(vec_1, base_point)
    vec_2 = vec_2 - metric.inner_product(vec_1, vec_2, base_point) * vec_1
    vec_2 /= metric.norm(vec_2, base_point)
    expected = bures_wasserstein_sectional_curvature(vec_1, vec_2, base_point)
    assert np.abs(result - expected) < tol

    vec_1_prime = P.T @ vec_1 @ P
    vec_2_prime = P.T @ vec_2 @ P
    vec_1_0_prime = np.zeros((dim, dim))
    vec_2_0_prime = np.zeros((dim, dim))
    coef = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            vec_1_0_prime[i, j] = vec_1_prime[i, j] / (d[i] + d[j])
            vec_2_0_prime[i, j] = vec_2_prime[i, j] / (d[i] + d[j])
            coef[i, j] = d[i] * d[j] / (d[i] + d[j])
    bracket = vec_1_0_prime @ vec_2_0_prime - vec_2_0_prime @ vec_1_0_prime
    expected = 3/2 * np.sum(coef * bracket ** 2)
    assert np.abs(result - expected) < tol


def test_bures_wasserstein_ricci_curvature_vectorization(tol=1e-5, n_points=10):
    dim = 2
    spd_space = SPDMatrices(dim)
    base_points = spd_space.random_point(n_points)
    vecs = spd_space.random_tangent_vec(base_points)
    result = bures_wasserstein_ricci_curvature(vecs, base_points)
    expected = np.array([bures_wasserstein_ricci_curvature(vc, bp) for (vc, bp) in zip(vecs, base_points)])
    assert np.all(np.abs(result - expected) < tol)


test_make_rotation_2d_vectorization()
test_projection_vectorization()
test_align_vectorization()
test_align()
test_bures_wasserstein_sectional_curvature()
test_bures_wasserstein_ricci_curvature_vectorization()
