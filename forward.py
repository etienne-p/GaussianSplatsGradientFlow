import numpy as np
from kernels import congruence, cov_world, schur_complement


def ortho_proj_ellipse_matrix(scale, rotation, view_matrix):
    """
    Orthographic projection of an ellipsoid; marginal covariance via slice.

    PARAMETERS
    scale       : (3,), semi-axes lengths (a, b, c)
    rotation    : (3, 3), rotation matrix of the ellipsoid
    view_matrix : (4, 4), world-to-view transform; projection drops view-space z

    RETURNS
    cov_proj : (2, 2), projected covariance matrix (marginal over depth)
    """
    R_v = view_matrix[:3, :3]  # (3, 3)
    cov = cov_world(rotation, scale)  # (3, 3)
    cov_v = congruence(R_v, cov)  # (3, 3), view-space covariance
    return cov_v[:2, :2]  # (2, 2), marginal: slice top-left 2x2 (drop depth)


def projected_gaussian_2d(pos, scale, rotation, view_matrix, use_schur=False):
    """
    Orthographic projection of a 3D Gaussian onto view's xy plane.

    PARAMETERS
    pos         : (3,), 3D position
    scale       : (3,), semi-axes lengths
    rotation    : (3, 3), rotation matrix
    view_matrix : (4, 4), world-to-view transform
    use_schur   : bool, if True use the Schur complement (conditional covariance xy|z=0)
                  rather than slicing the view-space covariance (marginal over depth)

    RETURNS
    cov_2d : (2, 2), projected 2D covariance
    mu_2d  : (2,), projected 2D mean
    """
    R_v = view_matrix[:3, :3]
    t_v = view_matrix[:3, 3]
    mu_2d = (R_v @ pos + t_v)[:2]
    cov = cov_world(rotation, scale)
    cov_v = congruence(R_v, cov)
    cov_2d = schur_complement(cov_v) if use_schur else cov_v[:2, :2]
    return cov_2d, mu_2d


def eval_gaussian_2d(precision, mu_2d, grid_x, grid_y):
    """
    Evaluate unnormalized 2D Gaussian exp(-0.5 q^T P q) on a meshgrid.

    PARAMETERS
    precision : (2, 2), precision matrix
    mu_2d     : (2,), mean
    grid_x    : (H, W), x-coordinates meshgrid
    grid_y    : (H, W), y-coordinates meshgrid

    RETURNS
    G     : (H, W), Gaussian values per pixel
    cache : tuple (q0, q1, G), saved intermediates for the backward pass
    """
    Q = np.stack([grid_x, grid_y], axis=-1) - mu_2d  # (H, W, 2)
    QM = Q @ precision  # (H, W, 2)
    QMQ = (QM * Q).sum(axis=-1)  # (H, W)
    G = np.exp(-0.5 * QMQ)  # (H, W)
    # Returns gaussian values, Xs and Ys, over a grid.
    return G, (Q[..., 0], Q[..., 1], G)


def compute_l1_error(pos1, scale1, R1, pos2, scale2, R2, view_matrix, grid_x, grid_y):
    """
    Project both Gaussians onto view_matrix's xy plane and compute |G1 - G2| on a grid.

    PARAMETERS
    pos1, scale1, R1 : Gaussian 1 parameters
    pos2, scale2, R2 : Gaussian 2 parameters
    view_matrix      : (4, 4), projection plane
    grid_x, grid_y   : (H, W), evaluation grid

    RETURNS
    diff : (H, W), signed pixel-wise difference G2 - G1
    l1   : float, total L1 error (integral of |diff| over the grid)
    """
    cov1, mu1 = projected_gaussian_2d(pos1, scale1, R1, view_matrix)  # (2,2), (2,)
    cov2, mu2 = projected_gaussian_2d(pos2, scale2, R2, view_matrix)  # (2,2), (2,)
    G1, _ = eval_gaussian_2d(np.linalg.inv(cov1), mu1, grid_x, grid_y)  # (H, W)
    G2, _ = eval_gaussian_2d(np.linalg.inv(cov2), mu2, grid_x, grid_y)  # (H, W)
    # We need the *signed* error
    diff = G2 - G1  # (H, W)
    # Compute the integral of the error over the grid
    dx = grid_x[0, 1] - grid_x[0, 0]
    dy = grid_y[1, 0] - grid_y[0, 0]
    l1 = np.abs(diff).sum() * dx * dy
    return diff, l1
