from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from numpy.linalg import norm
from pca.bures_wasserstein_principal_geodesic_analysis import BuresWassersteinPGA, BuresWassersteinPGAND
from pca.bures_wasserstein_tangent_pca import BuresWassersteinTPCA
from tools.compute import make_rotation_2d
from tools.generate import *


TOL = 1e-5


def check_horizontality(vec, point, tol=TOL):
    assert norm(point.T @ vec - vec.T @ point) / norm(point.T @ vec) < tol


def check_unit_norm(vec, tol=TOL):
    assert np.abs(norm(vec) - 1) < tol


def check_orthogonality(vec_1, vec_2, tol=TOL):
    assert np.abs(np.sum(vec_1 * vec_2)) < tol


def test__bures_wasserstein_pga__step1(dim=3, n_points=10, seed=123):
    """Check step 1 in dimension greater than 2."""
    rot_space = SpecialOrthogonal(dim)
    sym_space = SymmetricMatrices(dim)
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    np.random.seed(seed) if seed is not None else np.random.seed(np.random.randint(100))
    points_spd = spd_space.random_point(n_points)
    sq_roots = np.stack([sqrtm(pt_spd) for pt_spd in points_spd])
    rotations = rot_space.random_point(n_points)
    mean = np.random.rand(dim, dim)
    vec = sym_space.random_point() @ mean
    pga = BuresWassersteinPGA(dim)
    new_rotations = pga.component_k_step_1(rotations, sq_roots, mean, vec)
    assert np.all(rot_space.belongs(new_rotations))


def test__bures_wasserstein_pga__step2_dim2(n_points=40):
    """Check step 2 in dimension 2."""
    dim = 2
    sym_space = SymmetricMatrices(dim)
    points = np.random.rand(n_points, dim, dim)
    mean = np.random.rand(dim, dim)
    vec_1 = sym_space.random_point() @ mean
    pga = BuresWassersteinPGA(dim)
    new_mean, new_vec = pga.component_1_step_2(points, mean, vec_1)
    check_unit_norm(new_vec)
    check_horizontality(new_vec, new_mean)

    vec_2 = make_rotation_2d(np.pi / 2) @ vec_1
    check_orthogonality(vec_1, vec_2)
    angle, time = np.random.rand(2)
    new_vec_2, new_angle, new_time = pga.component_2_step_2(points, mean, vec_1, vec_2, angle, time)
    new_rotation = make_rotation_2d(new_angle)
    new_mean_2 = (mean + new_time * vec_1) @ new_rotation
    check_unit_norm(new_vec_2)
    check_orthogonality(new_vec_2, vec_1 @ new_rotation)
    check_horizontality(new_vec_2, new_mean_2)


def test__bures_wasserstein_pga__step2(dim=3, n_points=40):
    """Check step 2 in dimension greater than 2."""
    sym_space = SymmetricMatrices(dim)
    rot_space = SpecialOrthogonal(dim)
    points = np.random.rand(n_points, dim, dim)
    mean = np.random.rand(dim, dim)
    vec_1 = sym_space.random_point() @ mean
    vec_1 /= norm(vec_1)
    pga = BuresWassersteinPGA(dim)
    new_mean, new_vec = pga.component_1_step_2(points, mean, vec_1)
    check_unit_norm(new_vec)
    check_horizontality(new_vec, new_mean)

    vec_2 = sym_space.random_point() @ mean
    vec_2 = vec_2 - np.sum(vec_2 * vec_1) * vec_1
    vec_2 /= norm(vec_2)
    check_orthogonality(vec_1, vec_2)
    time = np.random.rand()
    rotation_2, old_rotation_2 = rot_space.random_point(2)
    new_vec_2, new_time = pga.component_2_step_2_2(rotation_2, old_rotation_2, time, vec_2, points, mean, vec_1)
    new_mean_2 = (mean + new_time * vec_1) @ rotation_2
    check_unit_norm(new_vec_2)
    check_orthogonality(new_vec_2, vec_1 @ rotation_2)
    check_horizontality(new_vec_2, new_mean_2)


def test__bures_wasserstein_pga__dim2(tol=1e-3, seed=123):
    """Check that implementations in dimension 2 and higher are consistent."""
    np.random.seed(seed)
    points_spd, _, _ = generate_spd_matrices_on_intersecting_geodesics(dim=2)
    mean_init, vec_init = generate_initialization(points_spd)
    res_pga_2d = BuresWassersteinPGA(dim=2, mean_init=mean_init, vec_1_init=vec_init).fit(points_spd)
    res_pga_nd = BuresWassersteinPGAND(mean_init=mean_init, vec_1_init=vec_init).fit(points_spd)
    result = np.array([res_pga_nd['cost_1'], res_pga_nd['cost_2']])
    expected = np.array([res_pga_2d['cost_1'], res_pga_2d['cost_2']])
    assert np.all(np.abs((result - expected) / expected) < tol)


def test__bures_wasserstein_pga__single_geodesic(dim=2, n_points=41, tol=1e-3, seed=123):
    """Check that geodesic projects to itself (component 1) and to its mean (component 2)."""
    times = np.linspace(0., 1., n_points)
    points_spd = generate_spd_matrices_on_geodesic(dim, times, seed)
    pga = BuresWassersteinPGA(dim).fit(points_spd)

    result = pga['mean_spd']
    expected = points_spd[n_points // 2]
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_1']
    expected = points_spd
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_2']
    expected = np.tile(pga['mean_spd'], (n_points, 1, 1))
    assert np.all(np.abs(result - expected) < tol)


def test__bures_wasserstein_pga__orthogonal_geodesics(dim=2, n_points=40, tol=1e-3, seed=123):
    """Check that two intersecting geodesics project to themselves."""
    times_1 = np.linspace(-0.5, 0.5, n_points // 2)
    times_2 = np.linspace(-0.3, 0.3, n_points // 2)
    points_spd, mean_spd = generate_spd_matrices_on_orthogonal_geodesics(dim, times_1, times_2, seed)
    pga = BuresWassersteinPGA(dim).fit(points_spd)

    result = pga['mean_spd']
    expected = mean_spd
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_1'][:n_points // 2]
    expected = points_spd[:n_points // 2]
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_1'][n_points // 2:]
    expected = np.tile(mean_spd, (n_points // 2, 1, 1))
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_2'][:n_points // 2]
    expected = np.tile(mean_spd, (n_points // 2, 1, 1))
    assert np.all(np.abs(result - expected) < tol)

    result = pga['component_2'][n_points // 2:]
    expected = points_spd[n_points // 2:]
    assert np.all(np.abs(result - expected) < tol)


def test__bures_wasserstein_pga__orthogonality(dim=2, tol=1e-5, seed=123):
    np.random.seed(seed)
    points_spd, _, _ = generate_spd_matrices_on_intersecting_geodesics(dim)
    mean_init, vec_init = generate_initialization(points_spd)
    res_pga = BuresWassersteinPGA(dim=dim, mean_init=mean_init, vec_1_init=vec_init).fit(points_spd)
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    result = spd_space.metric.inner_product(res_pga['vec_1_spd'], res_pga['vec_2_spd'], res_pga['mean_spd'])
    expected = 0.
    assert np.abs(result - expected) < tol


def test__bures_wasserstein_tpca__single_geodesic(dim=2, n_points=41, tol=1e-3, seed=123):
    """Check that geodesic projects to itself (component 1) and to its mean (component 2)."""
    times = np.linspace(0., 1., n_points)
    points_spd = generate_spd_matrices_on_geodesic(dim, times, seed)
    tpca = BuresWassersteinTPCA.fit(points_spd)

    result = tpca['mean_spd']
    expected = points_spd[n_points // 2]
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_1']
    expected = points_spd
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_2']
    expected = np.tile(tpca['mean_spd'], (n_points, 1, 1))
    assert np.all(np.abs(result - expected) < tol)


def test__bures_wasserstein_tpca__orthogonal_geodesics(dim=2, n_points=40, tol=1e-3, seed=123):
    """Check that two intersecting geodesics project to themselves."""
    times_1 = np.linspace(-0.5, 0.5, n_points // 2)
    times_2 = np.linspace(-0.3, 0.3, n_points // 2)
    points_spd, mean_spd = generate_spd_matrices_on_orthogonal_geodesics(dim, times_1, times_2, seed)
    tpca = BuresWassersteinTPCA.fit(points_spd)

    result = tpca['mean_spd']
    expected = mean_spd
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_1'][:n_points // 2]
    expected = points_spd[:n_points // 2]
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_1'][n_points // 2:]
    expected = np.tile(mean_spd, (n_points // 2, 1, 1))
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_2'][:n_points // 2]
    expected = np.tile(mean_spd, (n_points // 2, 1, 1))
    assert np.all(np.abs(result - expected) < tol)

    result = tpca['component_2'][n_points // 2:]
    expected = points_spd[n_points // 2:]
    assert np.all(np.abs(result - expected) < tol)


test__bures_wasserstein_pga__step1(dim=3)
test__bures_wasserstein_pga__step2(dim=3)
test__bures_wasserstein_pga__step2_dim2()
test__bures_wasserstein_pga__dim2()
test__bures_wasserstein_pga__single_geodesic(dim=2)
test__bures_wasserstein_pga__single_geodesic(dim=3)
test__bures_wasserstein_pga__orthogonal_geodesics(dim=2)
test__bures_wasserstein_pga__orthogonal_geodesics(dim=3)
test__bures_wasserstein_pga__orthogonality(dim=2)
test__bures_wasserstein_pga__orthogonality(dim=3)
test__bures_wasserstein_tpca__single_geodesic(dim=3)
test__bures_wasserstein_tpca__orthogonal_geodesics(dim=3)