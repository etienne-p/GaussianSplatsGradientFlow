import numpy as np


def gaussian_kernel(P, q):
    """
    Unnormalized Gaussian at a single point.

    PARAMETERS
    P : (2, 2), precision matrix
    q : (2,), displacement from mean

    RETURNS
    float, exp(-0.5 * q^T P q)
    """
    return np.exp(-0.5 * (q @ P @ q))


def congruence(R, X):
    """
    Congruence transformation R X R^T.

    PARAMETERS
    R : (N, N), transformation matrix
    X : (N, N), matrix to transform

    RETURNS
    (N, N), R @ X @ R.T
    """
    return R @ X @ R.T


def cov_world(R, s):
    """
    World-space covariance of an axis-aligned Gaussian scaled by s and rotated by R.

    PARAMETERS
    R : (3, 3), rotation matrix
    s : (3,), scale (semi-axes)

    RETURNS
    (3, 3), R diag(s^2) R^T
    """
    return R @ np.diag(s**2) @ R.T


def schur_complement(A, b, d):
    """
    Schur complement of d in the 3x3 block [[A, b], [b^T, d]].

    PARAMETERS
    A : (2, 2)
    b : (2,)
    d : float

    RETURNS
    (2, 2), A - b b^T / d
    """
    return A - np.outer(b, b) / d


def inv_grad(P, G):
    """
    Gradient of a scalar loss through matrix inversion: dL/dA given P = inv(A) and G = dL/dP.

    PARAMETERS
    P : (N, N), inv(A)
    G : (N, N), dL/dP

    RETURNS
    (N, N), -P G P
    """
    return -P @ G @ P

_4PI = 4 * np.pi

# Normalization constants for Cartesian real SH Y_l^m, keyed by (l, m).
# Derived from sqrt(rational/pi); add entries here to support higher degrees.
_SH_CONST: dict[tuple[int, int], float] = {
    (0,  0): np.sqrt( 1 / _4PI),
    (1, -1): np.sqrt( 3 / _4PI),
    (1,  0): np.sqrt( 3 / _4PI),
    (1,  1): np.sqrt( 3 / _4PI),
    (2, -2): np.sqrt(15 / _4PI),
    (2, -1): np.sqrt(15 / _4PI),
    (2,  0): np.sqrt( 5 / _4PI) / 2,
    (2,  1): np.sqrt(15 / _4PI),
    (2,  2): np.sqrt(15 / _4PI) / 2,
}


def sh_lm_index(l, m):
    """Flat index for real SH coefficient (l, m) in standard order: l^2 + l + m."""
    return l * l + l + m


def _iter_lm(n):
    """Yield (l, m, flat_index) for the first n SH coefficients in ascending l, m order."""
    i, l = 0, 0
    while i < n:
        for m in range(-l, l + 1):
            if i < n:
                yield l, m, i
                i += 1
        l += 1


def sh_basis(l, m, d):
    """
    Real spherical harmonic Y_l^m evaluated at unit direction(s) d.

    PARAMETERS
    l : int, degree >= 0
    m : int, order with -l <= m <= l
    d : (..., 3), unit direction(s)

    RETURNS
    (...,), Y_l^m(d)
    """
    c = _SH_CONST[(l, m)]
    if l == 0:
        return np.full(d.shape[:-1], c)
    if l == 1:
        if m == -1: return c * d[..., 1]  # ~ y
        if m ==  0: return c * d[..., 2]  # ~ z
        if m ==  1: return c * d[..., 0]  # ~ x
    if l == 2:
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        if m == -2: return c * x * y
        if m == -1: return c * y * z
        if m ==  0: return c * (2*z*z - x*x - y*y)
        if m ==  1: return c * x * z
        if m ==  2: return c * (x*x - y*y)
    raise NotImplementedError(f"sh_basis not implemented for l={l}, m={m}")


def sh_eval(sh_coeffs, d):
    """
    Evaluate SH color from a coefficient array at direction(s) d.

    PARAMETERS
    sh_coeffs : (n, 3), SH coefficients; row i corresponds to (l, m) from sh_lm_index
    d         : (..., 3), unit direction(s)

    RETURNS
    (..., 3), raw color (0.5 DC offset applied, no clip — clip at display boundary)
    """
    result = np.zeros(d.shape[:-1] + (3,))
    for l, m, i in _iter_lm(sh_coeffs.shape[0]):
        result += sh_basis(l, m, d)[..., np.newaxis] * sh_coeffs[i]
    return result + 0.5


def sh_grad(sh_coeffs, d, dL_dcolor):
    """
    Gradient of loss w.r.t. sh_coeffs for a single viewing direction d.

    PARAMETERS
    sh_coeffs : (n, 3), SH coefficients
    d         : (3,), unit direction
    dL_dcolor : (3,), dL/dcolor

    RETURNS
    (n, 3), dL/d_sh_coeffs
    """
    grad = np.zeros_like(sh_coeffs)
    for l, m, i in _iter_lm(sh_coeffs.shape[0]):
        grad[i] = sh_basis(l, m, d) * dL_dcolor
    return grad


# TODO: remove and just use scipy.spatial?
def quat_to_matrix(q):
    """
    Convert unit quaternion to rotation matrix.

    PARAMETERS
    q : (4,), unit quaternion [x, y, z, w]

    RETURNS
    (3, 3), rotation matrix R(q)
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quat_grad_from_rot_grad(q, dL_dR):
    """
    Gradient of loss w.r.t. unit quaternion q, given dL/dR.

    PARAMETERS
    q     : (4,), unit quaternion [x, y, z, w]
    dL_dR : (3, 3), dL/dR

    RETURNS
    (4,), [dL/dx, dL/dy, dL/dz, dL/dw]
    """
    x, y, z, w = q
    dL_dw = 2 * (
        -dL_dR[0, 1] * z
        + dL_dR[0, 2] * y
        + dL_dR[1, 0] * z
        - dL_dR[1, 2] * x
        - dL_dR[2, 0] * y
        + dL_dR[2, 1] * x
    )
    dL_dx = 2 * (
        dL_dR[0, 1] * y
        + dL_dR[0, 2] * z
        + dL_dR[1, 0] * y
        - 2 * dL_dR[1, 1] * x
        - dL_dR[1, 2] * w
        + dL_dR[2, 0] * z
        + dL_dR[2, 1] * w
        - 2 * dL_dR[2, 2] * x
    )
    dL_dy = 2 * (
        -2 * dL_dR[0, 0] * y
        + dL_dR[0, 1] * x
        + dL_dR[0, 2] * w
        + dL_dR[1, 0] * x
        + dL_dR[1, 2] * z
        - dL_dR[2, 0] * w
        + dL_dR[2, 1] * z
        - 2 * dL_dR[2, 2] * y
    )
    dL_dz = 2 * (
        -2 * dL_dR[0, 0] * z
        - dL_dR[0, 1] * w
        + dL_dR[0, 2] * x
        + dL_dR[1, 0] * w
        - 2 * dL_dR[1, 1] * z
        + dL_dR[1, 2] * y
        + dL_dR[2, 0] * x
        + dL_dR[2, 1] * y
    )
    return np.array([dL_dx, dL_dy, dL_dz, dL_dw])
