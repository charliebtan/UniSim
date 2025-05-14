import numpy as np


SG_THRESH = 1000    # sub-graph threshold = 1000 atoms


def graph_cut(x, radius_min=8.0, radius_max=20.0, center=None):
    """
    :param x: coordinates of all atoms, (N, 3), unit: Angstrom
    :param radius_min: subgraph for computing grads within a sphere of radius_min
    :param radius_max: subgraph for constructing edges within a sphere of radius_max
    :return: max_indices, mask
    """
    N = x.shape[0]  # atoms
    center = np.random.randint(N) if center is None else center
    xc = x[center]

    dist = np.linalg.norm(x - xc, axis=-1)

    min_indices = np.where(dist < radius_min)[0]
    max_indices = np.where(dist < radius_max)[0]

    mask = np.isin(max_indices, min_indices).astype(np.compat.long)

    return max_indices, mask
