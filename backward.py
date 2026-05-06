import numpy as np
from kernels import congruence, inv_grad, quat_to_matrix, quat_grad_from_rot_grad


def grad_eval_gaussian_2d(cache, precision, dL_dG):
    """
    Backward pass through eval_gaussian_2d.

    PARAMETERS
    cache     : tuple (q0, q1, G) returned by eval_gaussian_2d
    precision : (2, 2), precision matrix used in the forward pass
    dL_dG     : (H, W), upstream gradient dL/dG per pixel

    RETURNS
    dL_d_precision : (2, 2)
    dL_d_mu        : (2,)
    """
    q0, q1, G = cache

    w = dL_dG * G  # (H, W)

    # dG/d(P_ij) = G * (-0.5) * q_i * q_j
    s = -0.5 * w  # (H, W)
    dL_d_precision = np.array(
        [
            [np.sum(s * q0 * q0), np.sum(s * q0 * q1)],
            [np.sum(s * q1 * q0), np.sum(s * q1 * q1)],
        ]
    )  # (2, 2)

    # dG/dmu_k = G * (P q)_k
    Pq0 = precision[0, 0] * q0 + precision[0, 1] * q1  # (H, W)
    Pq1 = precision[1, 0] * q0 + precision[1, 1] * q1  # (H, W)
    dL_d_mu = np.array([np.sum(w * Pq0), np.sum(w * Pq1)])  # (2,)

    return dL_d_precision, dL_d_mu


def grad_projected_gaussian_2d(
    precision, scale, q, view_matrix, dL_d_precision, dL_d_mu
):
    """
    Backward pass through projected_gaussian_2d.

    Traces through: precision=inv(cov_proj), cov_proj=cov_v[:2,:2],
    cov_v=R_v cov R_v^T, cov=R(q) diag(s^2) R(q)^T.

    PARAMETERS
    precision      : (2, 2), inv(cov_proj) from the forward pass
    scale          : (3,), semi-axes lengths
    q              : (4,), unit quaternion [w, x, y, z]
    view_matrix    : (4, 4), world-to-view transform
    dL_d_precision : (2, 2), upstream gradient w.r.t. precision
    dL_d_mu        : (2,), upstream gradient w.r.t. projected mean

    RETURNS
    dL_d_pos   : (3,)
    dL_d_scale : (3,)
    dL_d_q     : (4,), gradient w.r.t. quaternion [w, x, y, z]
    """
    R = quat_to_matrix(q)  # (3, 3)
    R_v = view_matrix[:3, :3]  # (3, 3)

    # mu_2d = (R_v @ pos)[:2]
    dL_d_pos = R_v[:2, :].T @ dL_d_mu  # (3,)

    # precision = inv(cov_proj)  -> dL/dcov_proj = -P dL/dP P
    dL_d_cov_proj = inv_grad(precision, dL_d_precision)  # (2, 2)

    # cov_proj = cov_v[:2, :2]  -> pad gradient to 3x3
    dL_d_cov_v = np.zeros((3, 3))  # (3, 3)
    dL_d_cov_v[:2, :2] = dL_d_cov_proj

    # cov_v = R_v @ cov @ R_v^T  -> dL/dcov = R_v^T dL/dcov_v R_v
    dL_d_cov = congruence(R_v.T, dL_d_cov_v)  # (3, 3)

    # cov = R(q) @ diag(s^2) @ R(q)^T  -> dL/dR, then push through dR/dq
    D = np.diag(scale**2)  # (3, 3)
    H = R.T @ dL_d_cov @ R  # (3, 3)
    dL_d_scale = 2 * scale * np.diag(H)  # (3,)
    dL_d_R = 2 * dL_d_cov @ R @ D  # (3, 3)
    dL_d_q = quat_grad_from_rot_grad(q, dL_d_R)  # (4,)

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
