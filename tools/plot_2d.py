import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from tools.compute import compute_geodesic


def plot_spd_matrix(mat, **kwargs):
    angles = np.linspace(0., 2 * np.pi, 100)
    vecs = np.stack([np.cos(angles), np.sin(angles)])
    res = mat @ vecs
    plt.plot(res[0], res[1], **kwargs)


def plot_spd_matrix_translated(mat, vec, **kwargs):
    angles = np.linspace(0., 2 * np.pi, 100)
    vecs = np.stack([np.cos(angles), np.sin(angles)])
    res = mat @ vecs
    plt.plot(res[0] + vec[0], res[1] + vec[1], **kwargs)


def plot_spd_matrices(points_spd, figsize=(3, 3)):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)

    plt.figure(figsize=figsize)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color=map(i))
    plt.axis('equal')
    plt.show()


def plot_pca_results_with_ellipses(points_spd, pca, title='', fontsize=8, alpha=0.2):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 3))
    fig.add_subplot(141)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color=map(i))
    plt.axis('equal')
    plt.title('Initial points')

    fig.add_subplot(142)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        plot_spd_matrix(pca.components[0, i], color=map(i))
    plt.axis('equal')
    plt.title('Projections on 1st component')

    fig.add_subplot(143)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        plot_spd_matrix(pca.components[1, i], color=map(i))
    plt.axis('equal')
    plt.title('Projections on 2nd component')

    fig.add_subplot(144)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        plot_spd_matrix(pca.components[2, i], color=map(i))
    plt.axis('equal')
    plt.title('Projections on 3rd component')
    fig.suptitle(title)
    plt.show()


def plot_pca_results_with_ellipses_on_grid(points_spd, pca, n, n_transl=26):
    dim = pca.points_spd.shape[1]
    x = np.linspace(0., n_transl, n)
    xx = np.meshgrid(x, x)
    positions = np.stack(xx).T.reshape((n ** 2, dim))
    positions_comp1 = np.tile(np.stack((x, n_transl // 2 * np.ones(n))).T, (1,n)).reshape((n**2, 2))
    positions_comp2 = np.tile(np.stack((n_transl // 2 * np.ones(n), x)).T, (n,1))
    for i in range(n ** 2):
        plot_spd_matrix_translated(points_spd[i], positions[i], color='k')
        plot_spd_matrix_translated(pca.components[0, i], positions_comp1[i], color='r')
        plot_spd_matrix_translated(pca.components[1, i], positions_comp2[i], color='b')
    plt.axis('off')
    plt.show()


def spectral_to_cone_coordinates(eig, angle):
    lbd, mu = eig
    a = (lbd + mu) / 2
    b = (lbd - mu) / 2 * (np.cos(angle) ** 2 - np.sin(angle) ** 2)
    c = (lbd - mu) * np.cos(angle) * np.sin(angle)
    return np.array([a, b, c])


def mat_to_cone_coordinates(sym_mat):
    eig, eigenvec = np.linalg.eigh(sym_mat)
    angle = np.arctan(- eigenvec[0, 1] / eigenvec[1, 1])
    return spectral_to_cone_coordinates(eig, angle)


def cone_coordinates_of_component(res, n_comp, eps=0.2):
    tmin = np.min(res.times[n_comp]) - eps
    tmax = np.max(res.times[n_comp]) + eps
    component_spd = compute_geodesic(res.mean_spd, res.vecs_spd[n_comp], t_min_clip=tmin, t_max_clip=tmax)
    component_cone = np.stack([mat_to_cone_coordinates(mat) for mat in component_spd])
    return component_cone


def plot_pca_results_in_cone(points_spd, res, bound=5., eps=0.2, elev=30, azim=45):
    colors = ['red', 'blue', 'green']
    x = np.linspace(-bound, bound, 100)
    xx, yy = np.meshgrid(x, x)
    zz = np.sqrt(xx ** 2 + yy ** 2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, alpha=0.5)
    points_cone = np.stack([mat_to_cone_coordinates(mat) for mat in points_spd])
    ax.scatter(points_cone[:, 1], points_cone[:, 2], points_cone[:, 0], color='k', s=40)
    for i in range(3):
        component_cone = cone_coordinates_of_component(res, i, eps)
        ax.plot(component_cone [:, 1], component_cone [:, 2], component_cone [:, 0], color=colors[i], linewidth=2)
    ax.view_init(elev=elev, azim=azim)
    plt.show()


def compare_pca_components_in_cone(n_comp, points_spd, res_gpca, res_tpca, bound=5., eps=0.2):
    gpca_component = cone_coordinates_of_component(res_gpca, n_comp, eps)
    tpca_component = cone_coordinates_of_component(res_tpca, n_comp, eps)
    points = np.stack([mat_to_cone_coordinates(mat) for mat in points_spd])
    tpca_mean = mat_to_cone_coordinates(res_tpca.mean_spd)

    x = np.linspace(-bound, bound, 100)
    xx, yy = np.meshgrid(x, x)
    zz = np.sqrt(xx ** 2 + yy ** 2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, alpha=0.5)
    ax.scatter(points[:, 1], points[:, 2], points[:, 0], color='black', s=40)
    ax.scatter(tpca_mean[1], tpca_mean[2], tpca_mean[0], color='magenta')
    ax.plot(gpca_component[:, 1], gpca_component[:, 2], gpca_component[:, 0], color='red', linewidth=2)
    ax.plot(tpca_component[:, 1], tpca_component[:, 2], tpca_component[:, 0], '--', color='red', linewidth=2)
    #ax.set_xticks([-0.5, 0, 0.5])
    #ax.set_yticks([-0.5, 0, 0.5])
    #ax.set_zticks([0, 0.5])
    plt.show()


def compare_first_pca_components_in_section_of_cone(points_spd, res_gpca, res_tpca, eps=0.2):
    font = {'size': 16}
    matplotlib.rc('font', **font)
    n_comp = 0
    gpca_component = cone_coordinates_of_component(res_gpca, n_comp, eps)
    tpca_component = cone_coordinates_of_component(res_tpca, n_comp, eps)
    points = np.stack([mat_to_cone_coordinates(mat) for mat in points_spd])
    tpca_mean = mat_to_cone_coordinates(res_tpca.mean_spd)

    plt.figure()
    plt.scatter(points[:, 1], points[:, 2], color='black')
    plt.plot(gpca_component[:, 1], gpca_component[:, 2], '-', color='red', linewidth=3)
    plt.plot(tpca_component[:, 1], tpca_component[:, 2], '--', color='red', linewidth=3)
    plt.scatter(tpca_mean[1], tpca_mean[2], color='magenta', s=100)
    plt.axis('equal')
    plt.axis('off')
    plt.show()