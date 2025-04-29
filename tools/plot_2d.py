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


def plot_results(points_spd, res, title='', fontsize=8):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 3))
    fig.add_subplot(141)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color=map(i))
    plt.axis('equal')
    plt.title('initial points')

    fig.add_subplot(142)
    for i in range(int(n_points)):
        plot_spd_matrix(points_spd[i], color=map(i))
    for mat in res['components'][0]:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.axis('equal')
    plt.title('projections on 1st component')

    fig.add_subplot(143)
    for i in range(int(n_points)):
        plot_spd_matrix(points_spd[i], color=map(i))
    for mat in res['components'][1]:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.axis('equal')
    plt.title('projections on 2nd component')

    fig.add_subplot(144)
    for i in range(int(n_points)):
        plot_spd_matrix(points_spd[i], color=map(i))
    for mat in res['components'][2]:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.axis('equal')
    plt.title('projections on 3rd component')
    fig.suptitle(title)
    plt.show()


def plot_results_1(points_spd, res, title='', fontsize=8, alpha=0.2):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 3))
    fig.add_subplot(141)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color=map(i))
    plt.axis('equal')
    #plt.title('initial points')

    fig.add_subplot(142)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        #plot_spd_matrix(res['components'][0, i], color=map(i))
        plot_spd_matrix(res.components[0, i], color=map(i))
    plt.axis('equal')
    #plt.title('projections on 1st component')

    fig.add_subplot(143)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        #plot_spd_matrix(res['components'][1, i], color=map(i))
        plot_spd_matrix(res.components[1, i], color=map(i))
    plt.axis('equal')
    #plt.title('projections on 2nd component')

    fig.add_subplot(144)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color='grey', alpha=alpha)
    for i in range(int(n_points)):
        plot_spd_matrix(res.components[2, i], color=map(i))
        #plot_spd_matrix(res['components'][2, i], color=map(i))
    plt.axis('equal')
    #plt.title('projections on 3rd component')
    fig.suptitle(title)
    plt.show()


def plot_results_compare(points_spd, res_pga, res_tpca):
    n_points = points_spd.shape[0]
    n_half = int(n_points / 2)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(221)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pga['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA 1st component')

    fig.add_subplot(222)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pga['component_2']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA 2nd component')

    fig.add_subplot(223)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_tpca['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('TPCA 1st component')

    fig.add_subplot(224)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_tpca['component_2']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('TPCA 2nd component')
    plt.show()


def eig_angle_to_cone_coordinates(eig, angle):
    lbd, mu = eig
    a = (lbd + mu) / 2
    b = (lbd - mu) / 2 * (np.cos(angle) ** 2 - np.sin(angle) ** 2)
    c = (lbd - mu) * np.cos(angle) * np.sin(angle)
    return np.array([a, b, c])


def mat_to_cone_coordinates(sym_mat):
    eig, eigenvec = np.linalg.eigh(sym_mat)
    angle = np.arctan(- eigenvec[0, 1] / eigenvec[1, 1])
    return eig_angle_to_cone_coordinates(eig, angle)


def plot_results_on_cone(points_spd, res, bound=5., t_clip=2.):
    geod1 = compute_geodesic(res['mean_spd'], res['vecs_spd'][0], t_min_clip=-t_clip, t_max_clip=t_clip)
    geod2 = compute_geodesic(res['mean_spd'], res['vecs_spd'][1], t_min_clip=-t_clip, t_max_clip=t_clip)
    geod3 = compute_geodesic(res['mean_spd'], res['vecs_spd'][2], t_min_clip=-t_clip, t_max_clip=t_clip)
    coords_comp1 = np.stack([mat_to_cone_coordinates(mat) for mat in res['components'][0]])
    coords_comp2 = np.stack([mat_to_cone_coordinates(mat) for mat in res['components'][1]])
    coords_comp3 = np.stack([mat_to_cone_coordinates(mat) for mat in res['components'][2]])
    coords_geod1 = np.stack([mat_to_cone_coordinates(mat) for mat in geod1])
    coords_geod2 = np.stack([mat_to_cone_coordinates(mat) for mat in geod2])
    coords_geod3 = np.stack([mat_to_cone_coordinates(mat) for mat in geod3])
    coords_spd = np.stack([mat_to_cone_coordinates(mat) for mat in points_spd])

    x = np.linspace(-bound, bound, 100)
    xx, yy = np.meshgrid(x, x)
    zz = np.sqrt(xx ** 2 + yy ** 2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, alpha=0.5)
    ax.scatter(coords_spd[:, 1], coords_spd[:, 2], coords_spd[:, 0], color='black', s=40)
    #ax.scatter(coords_comp1[:, 1], coords_comp1[:, 2], coords_comp1[:, 0], color='red')
    #ax.scatter(coords_comp2[:, 1], coords_comp2[:, 2], coords_comp2[:, 0], color='blue')
    #ax.scatter(coords_comp3[:, 1], coords_comp3[:, 2], coords_comp3[:, 0], color='green')
    ax.plot(coords_geod1[:, 1], coords_geod1[:, 2], coords_geod1[:, 0], color='red', linewidth=2)
    ax.plot(coords_geod2[:, 1], coords_geod2[:, 2], coords_geod2[:, 0], color='blue', linewidth=2)
    ax.plot(coords_geod3[:, 1], coords_geod3[:, 2], coords_geod3[:, 0], color='green', linewidth=2)
    plt.show()
