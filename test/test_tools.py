from tools.compute import *


def test_vectorizations(tol=1e-5):
    angles = np.random.rand(5)
    result = make_rotation_2d(angles)
    expected = np.stack([make_rotation_2d(angle) for angle in angles])
    assert np.all(np.abs(result - expected) < tol)

    mats = np.random.rand(5, 2, 2)
    result = project(mats)
    expected = np.stack([project(mat) for mat in mats])
    assert np.all(np.abs(result - expected) < tol)

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


test_vectorizations()
test_align()