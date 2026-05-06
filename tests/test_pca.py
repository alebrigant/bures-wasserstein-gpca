from pca.bures_wasserstein_geodesic_pca import *
from pca.bures_wasserstein_tangent_pca import *
from tools.generate import *


TOL = 1e-5


def check_horizontality(vec, point, tol=TOL):
    assert norm(point.T @ vec - vec.T @ point) / norm(point.T @ vec) < tol


def check_unit_norm(vec, tol=TOL):
    assert np.abs(norm(vec) - 1) < tol


def check_orthogonality(vec_1, vec_2, tol=TOL):
    assert np.abs(np.sum(vec_1 * vec_2)) < tol


def test_clip_times(dim=3, tol=1e-5):
    n_points = 10
    points = np.random.rand(n_points, dim, dim)
    mean = np.random.rand(dim, dim)
    vec_aux = np.random.rand(dim, dim)
    vec = (vec_aux + vec_aux.T) @ mean
    result = []
    for point in points:
        result.append(clip_times(point, mean, vec)[0])
    result = np.stack(result)
    expected = clip_times(points, mean, vec)[0]
    assert np.all(np.abs(result - expected) < tol)


def test__bures_wasserstein_gpca__general(dim=2, tol=1e-5, seed=1):
    np.random.seed(seed)
    points_spd = generate_spd_matrices_on_intersecting_geodesics(dim, eps=0.2)[0]
    res = BuresWassersteinGPCA(dim=dim).fit(points_spd)
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    rdim = dim * (dim + 1) // 2
    norms = []
    inner_prods = []
    for i in range(rdim):
        norms.append(spd_space.metric.norm(res.vecs_spd[i], res.mean_spd))
        for j in range(i+1, rdim):
            inner_prods.append(spd_space.metric.inner_product(res.vecs_spd[i], res.vecs_spd[j], res.mean_spd))
    norms = np.stack(norms)
    inner_prods = np.stack(inner_prods)
    assert np.all(np.abs(norms - 1) < tol)
    assert np.all(np.abs(inner_prods) < tol)

    for i in range(rdim):
        check_horizontality(res.vec[i], res.mean[i])


def test__bures_wasserstein_gpca__dim2(tol=1e-3, seed=123):
    """Check that implementations in dimension 2 and higher are consistent."""
    np.random.seed(seed)
    points_spd = generate_spd_matrices_on_intersecting_geodesics(dim=2)[0]
    res_2d = BuresWassersteinGPCA(dim=2).fit(points_spd)
    res_nd = BuresWassersteinGPCAND(dim=2).fit(points_spd)
    result = res_nd.costs
    expected = res_2d.costs
    assert np.all(np.abs((result - expected) / expected) < tol)


def test__bures_wasserstein_gpca__orthogonal_geodesics(dim=2, n_points=11, n_geod=3, tol=1e-2, seed=1):
    """Check that multiple orthogonal geodesics project to themselves."""
    points_spd, mean_spd = generate_spd_matrices_on_orthogonal_geodesics(dim, n_points, n_geod, seed)
    pga = BuresWassersteinGPCA(dim).fit(points_spd)

    indices = np.arange(n_geod * n_points)
    masks = [(indices < n_points)]
    for i in range(1, n_geod - 1):
        masks.append((indices >= i * n_points) * (indices < (i+1) * n_points))
    masks.append((indices >= (n_geod - 1) * n_points))

    result = pga.mean_spd
    expected = mean_spd
    assert np.all(np.abs(result - expected) < tol)

    for i in range(n_geod):
        result = pga.components[i, masks[i]]
        expected = points_spd[masks[i]]
        assert np.all(np.abs(result - expected) < tol)

        result = pga.components[i, ~masks[i]]
        expected = np.tile(mean_spd, ((n_geod - 1) * n_points, 1, 1))
        assert np.all(np.abs(result - expected) < tol)


def test__bures_wasserstein_tpca__general(dim=2, tol=1e-6, seed=1):
    np.random.seed(seed)
    points_spd = generate_spd_matrices_on_intersecting_geodesics(dim, eps=0.2)[0]
    res = BuresWassersteinTPCA().fit(points_spd)
    spd_space = SPDMatrices(dim)
    spd_space.equip_with_metric(SPDBuresWassersteinMetric)
    rdim = dim * (dim + 1) // 2
    norms = []
    inner_prods = []
    for i in range(rdim):
        norms.append(spd_space.metric.norm(res.vecs_spd[i], res.mean_spd))
        for j in range(i+1, rdim):
            inner_prods.append(spd_space.metric.inner_product(res.vecs_spd[i], res.vecs_spd[j], res.mean_spd))
    norms = np.stack(norms)
    inner_prods = np.stack(inner_prods)
    assert np.all(np.abs(norms - 1) < tol)
    assert np.all(np.abs(inner_prods) < tol)


def test__bures_wasserstein_tpca__orthogonal_geodesics(dim=2, n_points=11, n_geod=3, tol=1e-2, seed=1):
    """Check that multiple orthogonal geodesics project to themselves."""
    points_spd, mean_spd = generate_spd_matrices_on_orthogonal_geodesics(dim, n_points, n_geod, seed)
    tpca = BuresWassersteinTPCA().fit(points_spd)

    indices = np.arange(n_geod * n_points)
    masks = [(indices < n_points)]
    for i in range(1, n_geod - 1):
        masks.append((indices >= i * n_points) * (indices < (i+1) * n_points))
    masks.append((indices >= (n_geod - 1) * n_points))

    result = tpca.mean_spd
    expected = mean_spd
    assert np.all(np.abs(result - expected) < tol)

    for i in range(n_geod):
        result = tpca.components[i, masks[i]]
        expected = points_spd[masks[i]]
        assert np.all(np.abs(result - expected) < tol)

        result = tpca.components[i, ~masks[i]]
        expected = np.tile(mean_spd, ((n_geod - 1) * n_points, 1, 1))
        assert np.all(np.abs(result - expected) < tol)


test_clip_times(dim=3)
test__bures_wasserstein_gpca__general(dim=2)
test__bures_wasserstein_gpca__general(dim=3)
test__bures_wasserstein_gpca__dim2()
test__bures_wasserstein_gpca__orthogonal_geodesics(dim=2, n_geod=1)
test__bures_wasserstein_gpca__orthogonal_geodesics(dim=2, n_geod=2)
test__bures_wasserstein_gpca__orthogonal_geodesics(dim=2, n_geod=3)
test__bures_wasserstein_gpca__orthogonal_geodesics(dim=3, tol=0.05)

test__bures_wasserstein_tpca__general(dim=2)
test__bures_wasserstein_tpca__general(dim=3)
test__bures_wasserstein_tpca__orthogonal_geodesics(dim=2, n_geod=1)
test__bures_wasserstein_tpca__orthogonal_geodesics(dim=2, n_geod=2)
test__bures_wasserstein_tpca__orthogonal_geodesics(dim=2, n_geod=3)
test__bures_wasserstein_tpca__orthogonal_geodesics(dim=3)
