import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_spd_matrix(mat, **kwargs):
    angles = np.linspace(0., 2 * np.pi, 100)
    vecs = np.stack([np.cos(angles), np.sin(angles)])
    res = mat @ vecs
    plt.plot(res[0], res[1], **kwargs)


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
