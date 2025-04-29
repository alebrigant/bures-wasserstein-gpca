from tools.compute import *
from tools.generate import generate_spd_matrices_on_orthogonal_geodesics
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.spd_matrices import SPDMatrices


def test_gram_schmidt(dim=5, tol=1e-5):
    n_vecs = 5
    spd_space = SPDMatrices(dim)
    metric = spd_space.random_point()
    vecs = np.random.rand(n_vecs, dim)
    vecs_new = gram_schmidt(vecs, metric)
    assert np.all(np.abs(vecs_new @ metric @ vecs_new.T - np.eye(dim)) < tol)


def test_from_vec_to_sym():
    vec = np.arange(6)
    dim = 3
    result = from_vec_to_sym(vec, dim)
    expected = np.array([
        [0, 1, 2], [1, 3, 4], [2, 4, 5]
    ])
    assert np.all(result == expected)

    result = from_sym_to_vec(expected, dim)
    expected = vec
    assert np.all(result == expected)


def test_make_rotation_2d_vectorization(tol=1e-5):
    angles = np.random.rand(5)
    result = make_rotation_2d(angles)
    expected = np.stack([make_rotation_2d(angle) for angle in angles])
    assert np.all(np.abs(result - expected) < tol)


def test_project_vectorization(tol=1e-5):
    mats = np.random.rand(5, 2, 2)
    result = project(mats)
    expected = np.stack([project(mat) for mat in mats])
    assert np.all(np.abs(result - expected) < tol)


def test_tangent_project_vectorization(tol=1e-5):
    vecs = np.random.rand(5, 2, 2)
    mat = np.random.rand(2, 2)
    result = tangent_project(vecs, mat)
    expected = np.stack([tangent_project(vec, mat) for vec in vecs])
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


def test_generate_spd_matrices_on_orthogonal_geodesics(dim=2, n_geod=3, tol=1e-5):
    """Check that geodesics are orthogonal and of decreasing length."""
    n_points = 21
    points_spd, mean_spd = generate_spd_matrices_on_orthogonal_geodesics(dim, n_points)
    space_spd = SPDMatrices(dim)
    space_spd.equip_with_metric(SPDBuresWassersteinMetric)
    geodesics = points_spd.reshape((n_geod, n_points, dim, dim))
    vecs_spd = np.stack([space_spd.metric.log(geodesics[n, -1], mean_spd) for n in range(n_geod)])
    inner_prods = []
    for i in range(n_geod):
        for j in range(i + 1, n_geod):
            inner_prods.append(space_spd.metric.inner_product(vecs_spd[i], vecs_spd[j], mean_spd))
    assert np.all(np.array(inner_prods) < tol)
    norms = space_spd.metric.norm(vecs_spd, mean_spd)
    indices = np.argsort(norms)[::-1]
    assert np.all(indices == np.arange(n_geod))


test_gram_schmidt()
test_from_vec_to_sym()
test_make_rotation_2d_vectorization()
test_project_vectorization()
test_tangent_project_vectorization()
test_align_vectorization()
test_align()
test_bures_wasserstein_sectional_curvature()
test_bures_wasserstein_ricci_curvature_vectorization()
test_generate_spd_matrices_on_orthogonal_geodesics()
