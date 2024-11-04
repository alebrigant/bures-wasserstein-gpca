import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_spd_matrix(mat, **kwargs):
    angles = np.linspace(0., 2 * np.pi, 100)
    vecs = np.stack([np.cos(angles), np.sin(angles)])
    res = mat @ vecs
    plt.plot(res[0], res[1], **kwargs) #color=color, marker=marker)
    # return res


def plot_results(points_spd, res, title='', mean=None, expected_proj=None):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)
    if expected_proj is not None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(121)
        for i in range(n_points):
            plot_spd_matrix(points_spd[i], color=map(i))
        plt.title('initial points')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        fig.add_subplot(122)
        for i in range(n_points):
            plot_spd_matrix(expected_proj[i], color='k', linestyle='dotted')
            plot_spd_matrix(res['component_1'][i], color=map(i))
        plt.title('projections vs expected projections (dotted)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
    else:
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(221)
        for i in range(n_points):
            plot_spd_matrix(points_spd[i], color=map(i))
        if mean is not None:
            plot_spd_matrix(mean, color='r', linewidth=4)
        plt.axis('equal')
        plt.title('initial points')

        fig.add_subplot(222)
        if mean is not None:
            plot_spd_matrix(mean, color='r')
        plot_spd_matrix(res['mean_spd'], color='b')
        plt.axis('equal')
        plt.title('mean vs true mean')

        fig.add_subplot(223)
        for i in range(int(n_points)): # / 2)):
            plot_spd_matrix(points_spd[i], color=map(i))
        for mat in res['component_1']:
            plot_spd_matrix(mat, color='k', linestyle='dotted')
        plt.axis('equal')
        plt.title('projections on 1st component')

        fig.add_subplot(224)
        for i in range(int(n_points)): # / 2), n_points):
            plot_spd_matrix(points_spd[i], color=map(i))
        for mat in res['component_2']:
            plot_spd_matrix(mat, color='k', linestyle='dotted')
        plt.axis('equal')
        plt.title('projections on 2nd component')
        fig.suptitle(title)
        plt.show()


def plot_results_compare(points_spd, res_pga, res_tpca):
    n_points = points_spd.shape[0]
    n_half = int(n_points / 2)

    fig = plt.figure(figsize=(10, 10))
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


def plot_results_compare_3(points_spd, res_pga2D, res_pgaND, res_tpca):
    n_points = points_spd.shape[0]
    n_half = int(n_points / 2)

    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(231)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pga2D['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA-2D component 1')

    fig.add_subplot(232)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pgaND['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA-ND component 1')

    fig.add_subplot(233)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_tpca['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('TPCA component 1')

    fig.add_subplot(234)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pga2D['component_2']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA-2D component 2')

    fig.add_subplot(235)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_pgaND['component_2']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('PGA-ND component 2')

    fig.add_subplot(236)
    for i in range(n_half):
        plot_spd_matrix(points_spd[i], color='b')
        plot_spd_matrix(points_spd[n_half + i], color='m')
    for mat in res_tpca['component_2']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('TPCA component 2')
    plt.show()


def plot_first_component(points_spd, res, title='', mean=None):
    n_points = points_spd.shape[0]
    map = matplotlib.colormaps['cool'].resampled(n_points)

    fig = plt.figure(figsize=(12, 4))
    fig.add_subplot(131)
    for i in range(n_points):
        plot_spd_matrix(points_spd[i], color=map(i))
    if mean is not None:
        plot_spd_matrix(mean, color='r', linewidth=4)
    plt.title('initial points')

    fig.add_subplot(132)
    if mean is not None:
        plot_spd_matrix(mean, color='r')
    plot_spd_matrix(res['mean_spd'], color='b')
    plt.title('mean vs true mean')

    fig.add_subplot(133)
    for i in range(int(n_points)): # / 2)):
        plot_spd_matrix(points_spd[i], color=map(i))
    for mat in res['component_1']:
        plot_spd_matrix(mat, color='k', linestyle='dotted')
    plt.title('projections on 1st component')
    fig.suptitle(title)
    plt.show()