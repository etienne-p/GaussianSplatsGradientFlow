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
    Transforming a matrix X by a change of basis R.

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


# The schur complement can be used to compute an exact projection of the gaussian from 3D to 2D.
# It would work for an orthographic projection. For a perspective projection,
# it's an approximation that ignores the depth dependence of the projection.
# That is, it ignores the vanishing of scale with depth.
def schur_complement(M):
    """
    Schur complement of the bottom-right element in a 3x3 block matrix.

    PARAMETERS
    M : (3, 3), symmetric matrix partitioned as [[A, b], [b^T, d]]

    RETURNS
    (2, 2), A - b b^T / d
    """
    b = M[:2, 2]
    d = M[2, 2]
    return M[:2, :2] - np.outer(b, b) / d


def grad_schur_complement(M, dL_dS):
    """
    Backprop through Schur complement S = A - outer(b, b) / d.

    PARAMETERS
    M     : (3, 3), the symmetric matrix [[A, b], [b^T, d]] from the forward pass
    dL_dS : (..., 2, 2), upstream gradient (scalar or batched)

    RETURNS
    (..., 3, 3), dL/dM
    """
    b = M[:2, 2]  # (2,)
    d = M[2, 2]  # scalar

    dL_dM = np.zeros((*dL_dS.shape[:-2], 3, 3))
    dL_dM[..., :2, :2] = dL_dS
    dL_dM[..., :2, 2] = -(dL_dS + dL_dS.swapaxes(-1, -2)) @ b / d  # (..., 2)
    dL_dM[..., 2, 2] = (dL_dS * np.outer(b, b)).sum(axis=(-2, -1)) / d**2  # (...)

    return dL_dM


def inv_grad(P, G):
    """
    Gradient of a scalar loss through matrix inversion: dL/dA given P = inv(A) and G = dL/dP.

    PARAMETERS
    P : (N, N), inv(A)
    G : (N, N), dL/dP

    RETURNS
    (N, N), -P G P
    """
    # Inverting a matrix reverses the direction of any perturbation to its input.
    # For us, P is the precision matrix, inverse of covariance, symmetric.
    return -P @ G @ P


_4PI = 4 * np.pi

# Normalization constants for Cartesian real SH Y_l^m, keyed by (l, m).
# Derived from sqrt(rational/pi); add entries here to support higher degrees.
_SH_CONST: dict[tuple[int, int], float] = {
    (0, 0): np.sqrt(1 / _4PI),
    (1, -1): np.sqrt(3 / _4PI),
    (1, 0): np.sqrt(3 / _4PI),
    (1, 1): np.sqrt(3 / _4PI),
    (2, -2): np.sqrt(15 / _4PI),
    (2, -1): np.sqrt(15 / _4PI),
    (2, 0): np.sqrt(5 / _4PI) / 2,
    (2, 1): np.sqrt(15 / _4PI),
    (2, 2): np.sqrt(15 / _4PI) / 2,
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
        if m == -1:
            return c * d[..., 1]  # ~ y
        if m == 0:
            return c * d[..., 2]  # ~ z
        if m == 1:
            return c * d[..., 0]  # ~ x
    if l == 2:
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        if m == -2:
            return c * x * y
        if m == -1:
            return c * y * z
        if m == 0:
            return c * (2 * z * z - x * x - y * y)
        if m == 1:
            return c * x * z
        if m == 2:
            return c * (x * x - y * y)
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


def quat_to_matrix(q):
    """
    Convert unit quaternion to rotation matrix.
    The matrix form of Rodrigues' rotation formula,
    R is a linear function of the outer products of q's components.

    PARAMETERS
    q : (4,), unit quaternion [x, y, z, w]

    RETURNS
    (3, 3), rotation matrix R(q)
    """
    x, y, z, w = q
    v = np.array([x, y, z])

    I = np.eye(3)
    vvT = np.outer(v, v)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    return (w**2 - v @ v) * I + 2 * vvT + 2 * w * skew


def quat_grad_from_rot_grad(q, dL_dR):
    """
    Gradient of loss w.r.t. unit quaternion q, given dL/dR.
    The gradient is a projection via Frobenius inner product onto each of
    the derivative matrices of the matrix form of Rodrigues' rotation formula.

    PARAMETERS
    q     : (4,), unit quaternion [x, y, z, w]
    dL_dR : (..., 3, 3), dL/dR — scalar (3,3) or any batch shape

    RETURNS
    (..., 4), [dL/dx, dL/dy, dL/dz, dL/dw]
    """
    x, y, z, w = q
    v = np.array([x, y, z])

    I = np.eye(3)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    dR_dw = 2 * w * I + 2 * skew

    dskew_dx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    dskew_dy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    dskew_dz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    def dR_dvi(ei, dskew_dvi):
        return (
            -2 * (v @ ei) * I
            + 2 * (np.outer(v, ei) + np.outer(ei, v))
            + 2 * w * dskew_dvi
        )

    dR_dx = dR_dvi(np.array([1, 0, 0]), dskew_dx)
    dR_dy = dR_dvi(np.array([0, 1, 0]), dskew_dy)
    dR_dz = dR_dvi(np.array([0, 0, 1]), dskew_dz)

    def frob(B):
        return (dL_dR * B).sum(axis=(-2, -1))

    return np.stack([frob(dR_dx), frob(dR_dy), frob(dR_dz), frob(dR_dw)], axis=-1)
