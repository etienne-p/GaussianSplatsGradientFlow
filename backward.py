import numpy as np
from kernels import congruence, inv_grad, quat_to_matrix, quat_grad_from_rot_grad


def grad_eval_gaussian_2d_per_pixel(cache, precision, dL_dG):
    """
    Per-pixel backward through eval_gaussian_2d — does NOT sum over pixels.

    PARAMETERS
    cache     : tuple (d0, d1, G) returned by eval_gaussian_2d
    precision : (2, 2), precision matrix used in the forward pass
    dL_dG     : (H, W), upstream gradient dL/dG per pixel

    RETURNS
    dL_d_precision : (H, W, 2, 2), per-pixel contribution to precision gradient
    dL_d_mu        : (H, W, 2),    per-pixel contribution to mu gradient
    """
    d0, d1, G = cache
    w = dL_dG * G   # (H, W)
    s = -0.5 * w    # (H, W)

    dL_d_precision = np.stack(
        [np.stack([s * d0 * d0, s * d0 * d1], axis=-1),
         np.stack([s * d1 * d0, s * d1 * d1], axis=-1)],
        axis=-2,
    )  # (H, W, 2, 2)

    Pq0 = precision[0, 0] * d0 + precision[0, 1] * d1  # (H, W)
    Pq1 = precision[1, 0] * d0 + precision[1, 1] * d1  # (H, W)
    dL_d_mu = w[..., None] * np.stack([Pq0, Pq1], axis=-1)  # (H, W, 2)

    return dL_d_precision, dL_d_mu


def grad_eval_gaussian_2d(cache, precision, dL_dG):
    """
    Backward pass through eval_gaussian_2d.

    PARAMETERS
    cache     : tuple (d0, d1, G) returned by eval_gaussian_2d
    precision : (2, 2), precision matrix used in the forward pass
    dL_dG     : (H, W), upstream gradient dL/dG per pixel

    RETURNS
    dL_d_precision : (2, 2)
    dL_d_mu        : (2,)
    """
    dL_d_precision, dL_d_mu = grad_eval_gaussian_2d_per_pixel(cache, precision, dL_dG)
    return dL_d_precision.sum(axis=(0, 1)), dL_d_mu.sum(axis=(0, 1))


def grad_projected_gaussian_2d(
    precision, scale, q, view_matrix, dL_d_precision, dL_d_mu
):
    """
    Backward pass through projected_gaussian_2d.

    Traces through: precision=inv(cov_proj), cov_proj=cov_v[:2,:2],
    cov_v=R_v cov R_v^T, cov=R(q) diag(s^2) R(q)^T.

    Accepts both scalar and batched upstream gradients:

    PARAMETERS
    precision      : (2, 2), inv(cov_proj) from the forward pass
    scale          : (3,), semi-axes lengths
    q              : (4,), unit quaternion [x, y, z, w]
    view_matrix    : (4, 4), world-to-view transform
    dL_d_precision : (..., 2, 2), upstream gradient w.r.t. precision
    dL_d_mu        : (..., 2),    upstream gradient w.r.t. projected mean

    RETURNS
    dL_d_pos   : (..., 3)
    dL_d_scale : (..., 3)
    dL_d_q     : (..., 4)
    """
    R = quat_to_matrix(q)  # (3, 3)
    R_v = view_matrix[:3, :3]  # (3, 3)

    # mu_2d = (R_v @ pos)[:2]  ->  dL/dpos = R_v[:2,:].T @ dmu
    # Written as dmu @ R_v[:2,:] so it broadcasts over any leading batch dims.
    dL_d_pos = dL_d_mu @ R_v[:2, :]  # (..., 3)

    # precision = inv(cov_proj)  -> dL/dcov_proj = -P dL/dP P
    dL_d_cov_proj = inv_grad(precision, dL_d_precision)  # (..., 2, 2)

    # cov_proj = cov_v[:2, :2]  -> pad gradient to 3x3
    dL_d_cov_v = np.zeros((*dL_d_cov_proj.shape[:-2], 3, 3))
    dL_d_cov_v[..., :2, :2] = dL_d_cov_proj

    # cov_v = R_v @ cov @ R_v^T  -> dL/dcov = R_v^T dL/dcov_v R_v
    dL_d_cov = congruence(R_v.T, dL_d_cov_v)  # (..., 3, 3)

    # cov = R(q) @ diag(s^2) @ R(q)^T  -> dL/dR, then push through dR/dq
    D = np.diag(scale**2)  # (3, 3)
    H = R.T @ dL_d_cov @ R  # (..., 3, 3)
    dL_d_scale = 2 * scale * np.einsum("...ii->...i", H)  # (..., 3)
    dL_d_R = 2 * dL_d_cov @ R @ D  # (..., 3, 3)
    dL_d_q = quat_grad_from_rot_grad(q, dL_d_R)  # (..., 4)

    return dL_d_pos, dL_d_scale, dL_d_q


def grad_gaussian_2d(cache_G, precision, scale, q, view_matrix, dL_dG):
    """
    Full backward pass from dL/dG to gradients w.r.t. Gaussian parameters.

    Chains grad_eval_gaussian_2d ->grad_projected_gaussian_2d.

    PARAMETERS
    cache_G     : tuple (q0, q1, G) returned by eval_gaussian_2d
    precision   : (2, 2), inv(cov_2d) used in the forward pass
    scale       : (3,), semi-axes lengths
    q           : (4,), unit quaternion [w, x, y, z]
    view_matrix : (4, 4), world-to-view transform
    dL_dG       : (H, W), upstream gradient dL/dG per pixel

    RETURNS
    dL_d_pos   : (3,)
    dL_d_scale : (3,)
    dL_d_q     : (4,)
    """
    dL_d_precision, dL_d_mu = grad_eval_gaussian_2d(cache_G, precision, dL_dG)
    return grad_projected_gaussian_2d(
        precision, scale, q, view_matrix, dL_d_precision, dL_d_mu
    )
