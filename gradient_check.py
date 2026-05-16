import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from forward import eval_gaussian_2d, projected_gaussian_2d
from backward import grad_eval_gaussian_2d, grad_projected_gaussian_2d
from kernels import (
    quat_to_matrix,
    quat_grad_from_rot_grad,
    sh_eval,
    sh_grad,
    schur_complement,
    grad_schur_complement,
)


def finite_diff_grad(f, x, eps=1e-5):
    """
    Central-difference numerical gradient of scalar f at point x.

    PARAMETERS
    f   : callable, scalar-valued function of x
    x   : ndarray, point at which to evaluate the gradient
    eps : float, finite-difference step size

    RETURNS
    grad : ndarray, same shape as x
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp, xm = x.copy().astype(float), x.copy().astype(float)
        xp.flat[i] += eps
        xm.flat[i] -= eps
        grad.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def rel_err(a, n):
    return np.max(np.abs(a - n)) / (np.max(np.abs(n)) + 1e-12)


class TestGradients(unittest.TestCase):
    eps_diff = 1e-5
    eps_thrs = 1e-8

    def test_grad_eval_gaussian_2d(self):
        rng = np.random.default_rng(42)

        xs = np.linspace(-3, 3, 20)
        ys = np.linspace(-3, 3, 20)
        grid_x, grid_y = np.meshgrid(xs, ys)

        scale = np.array([1.5, 1.0, 0.7])
        q = Rotation.random(random_state=42).as_quat()
        pos = np.array([0.1, -0.2, 0.0])
        view_matrix = np.eye(4)

        cov_2d, mu_2d = projected_gaussian_2d(
            pos, scale, quat_to_matrix(q), view_matrix
        )
        precision = np.linalg.inv(cov_2d)
        dL_dG = rng.standard_normal(grid_x.shape)

        _, cache_G = eval_gaussian_2d(precision, mu_2d, grid_x, grid_y)
        dL_dP_a, dL_dmu_a = grad_eval_gaussian_2d(cache_G, precision, dL_dG)

        dL_dP_n = finite_diff_grad(
            lambda P: (dL_dG * eval_gaussian_2d(P, mu_2d, grid_x, grid_y)[0]).sum(),
            precision,
            self.eps_diff,
        )
        dL_dmu_n = finite_diff_grad(
            lambda mu: (
                dL_dG * eval_gaussian_2d(precision, mu, grid_x, grid_y)[0]
            ).sum(),
            mu_2d,
            self.eps_diff,
        )

        print("grad_eval_gaussian_2d")
        print(f"  precision  rel err = {rel_err(dL_dP_a,  dL_dP_n):.2e}")
        print(f"  mu         rel err = {rel_err(dL_dmu_a, dL_dmu_n):.2e}")

        self.assertLess(
            rel_err(dL_dP_a, dL_dP_n),
            self.eps_thrs,
            f"precision rel err {rel_err(dL_dP_a, dL_dP_n):.2e} >= {self.eps_thrs}",
        )
        self.assertLess(
            rel_err(dL_dmu_a, dL_dmu_n),
            self.eps_thrs,
            f"mu rel err {rel_err(dL_dmu_a, dL_dmu_n):.2e} >= {self.eps_thrs}",
        )

    def test_grad_schur_complement(self):
        rng = np.random.default_rng(42)

        # Random symmetric positive-definite 3x3 matrix
        A = rng.standard_normal((3, 3))
        M = A @ A.T + np.eye(3)

        G = rng.standard_normal((2, 2))
        dL_dS = (G + G.T) / 2  # symmetric upstream gradient

        dL_dM_a = grad_schur_complement(M, dL_dS)
        dL_dM_n = finite_diff_grad(
            lambda m: (dL_dS * schur_complement(m)).sum(), M, self.eps_diff
        )

        print("grad_schur_complement")
        print(f"  M  rel err = {rel_err(dL_dM_a, dL_dM_n):.2e}")

        self.assertLess(
            rel_err(dL_dM_a, dL_dM_n),
            self.eps_thrs,
            f"M rel err {rel_err(dL_dM_a, dL_dM_n):.2e} >= {self.eps_thrs}",
        )

    def _check_grad_projected_gaussian_2d(self, use_schur):
        rng = np.random.default_rng(42)

        scale = np.array([1.5, 1.0, 0.7])
        q = Rotation.random(random_state=42).as_quat()
        pos = np.array([0.1, -0.2, 0.0])
        view_matrix = np.eye(4)

        cov_2d, _ = projected_gaussian_2d(
            pos, scale, quat_to_matrix(q), view_matrix, use_schur=use_schur
        )
        precision = np.linalg.inv(cov_2d)

        M = rng.standard_normal((2, 2))
        dL_d_precision = (M + M.T) / 2
        dL_d_mu = rng.standard_normal(2)

        def proj_loss(s=scale, q_=q, p=pos):
            cov, mu = projected_gaussian_2d(
                p, s, quat_to_matrix(q_), view_matrix, use_schur=use_schur
            )
            return (dL_d_precision * np.linalg.inv(cov)).sum() + (dL_d_mu * mu).sum()

        dL_d_pos_a, dL_d_scale_a, dL_d_q_a = grad_projected_gaussian_2d(
            precision,
            scale,
            q,
            view_matrix,
            dL_d_precision,
            dL_d_mu,
            use_schur=use_schur,
        )

        dL_d_scale_n = finite_diff_grad(lambda s: proj_loss(s=s), scale, self.eps_diff)
        dL_d_q_n = finite_diff_grad(lambda q_: proj_loss(q_=q_), q, self.eps_diff)
        dL_d_pos_n = finite_diff_grad(lambda p: proj_loss(p=p), pos, self.eps_diff)

        print(f"grad_projected_gaussian_2d(use_schur={use_schur})")
        print(f"  scale  rel err = {rel_err(dL_d_scale_a, dL_d_scale_n):.2e}")
        print(f"  q      rel err = {rel_err(dL_d_q_a,     dL_d_q_n):.2e}")
        print(f"  pos    rel err = {rel_err(dL_d_pos_a,   dL_d_pos_n):.2e}")

        self.assertLess(
            rel_err(dL_d_scale_a, dL_d_scale_n),
            self.eps_thrs,
            f"scale rel err {rel_err(dL_d_scale_a, dL_d_scale_n):.2e} >= {self.eps_thrs}",
        )
        self.assertLess(
            rel_err(dL_d_q_a, dL_d_q_n),
            self.eps_thrs,
            f"q rel err {rel_err(dL_d_q_a, dL_d_q_n):.2e} >= {self.eps_thrs}",
        )
        self.assertLess(
            rel_err(dL_d_pos_a, dL_d_pos_n),
            self.eps_thrs,
            f"pos rel err {rel_err(dL_d_pos_a, dL_d_pos_n):.2e} >= {self.eps_thrs}",
        )

    def test_grad_projected_gaussian_2d(self):
        for use_schur in [False, True]:
            with self.subTest(use_schur=use_schur):
                self._check_grad_projected_gaussian_2d(use_schur)

    def test_quat_grad_from_rot_grad(self):
        rng = np.random.default_rng(42)

        q = Rotation.random(random_state=42).as_quat()
        G = rng.standard_normal((3, 3))  # dL/dR

        dL_dq_a = quat_grad_from_rot_grad(q, G)
        dL_dq_n = finite_diff_grad(
            lambda q_: (G * quat_to_matrix(q_)).sum(), q, self.eps_diff
        )

        print("quat_grad_from_rot_grad")
        print(f"  q  rel err = {rel_err(dL_dq_a, dL_dq_n):.2e}")

        self.assertLess(
            rel_err(dL_dq_a, dL_dq_n),
            self.eps_thrs,
            f"q rel err {rel_err(dL_dq_a, dL_dq_n):.2e} >= {self.eps_thrs}",
        )

    def test_sh_grad_deg0(self):
        rng = np.random.default_rng(42)

        sh_coeffs = rng.standard_normal((1, 3))
        d = np.zeros(3)  # direction irrelevant for l=0
        dL_dcolor = rng.standard_normal(3)

        dL_d_sh_a = sh_grad(sh_coeffs, d, dL_dcolor)
        dL_d_sh_n = finite_diff_grad(
            lambda s: (dL_dcolor * sh_eval(s, d)).sum(),
            sh_coeffs,
            self.eps_diff,
        )

        print("sh_grad (deg 0)")
        print(f"  sh_coeffs  rel err = {rel_err(dL_d_sh_a, dL_d_sh_n):.2e}")

        self.assertLess(
            rel_err(dL_d_sh_a, dL_d_sh_n),
            self.eps_thrs,
            f"sh_coeffs rel err {rel_err(dL_d_sh_a, dL_d_sh_n):.2e} >= {self.eps_thrs}",
        )

    def test_sh_grad_deg01(self):
        rng = np.random.default_rng(42)

        sh_coeffs = np.vstack(
            [
                rng.uniform(-1.5, 1.5, (1, 3)),
                rng.uniform(-0.8, 0.8, (3, 3)),
            ]
        )
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        dL_dcolor = rng.standard_normal(3)

        dL_d_sh_a = sh_grad(sh_coeffs, d, dL_dcolor)
        dL_d_sh_n = finite_diff_grad(
            lambda s: (dL_dcolor * sh_eval(s, d)).sum(), sh_coeffs, self.eps_diff
        )

        print("sh_grad (deg 0+1)")
        print(f"  sh_coeffs  rel err = {rel_err(dL_d_sh_a, dL_d_sh_n):.2e}")

        self.assertLess(
            rel_err(dL_d_sh_a, dL_d_sh_n),
            self.eps_thrs,
            f"sh_coeffs rel err {rel_err(dL_d_sh_a, dL_d_sh_n):.2e} >= {self.eps_thrs}",
        )

    def test_sh_grad_deg012(self):
        rng = np.random.default_rng(42)

        sh_coeffs = np.vstack(
            [
                rng.uniform(-1.5, 1.5, (1, 3)),
                rng.uniform(-0.8, 0.8, (3, 3)),
                rng.uniform(-0.3, 0.3, (5, 3)),
            ]
        )
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        dL_dcolor = rng.standard_normal(3)

        dL_d_sh_a = sh_grad(sh_coeffs, d, dL_dcolor)
        dL_d_sh_n = finite_diff_grad(
            lambda s: (dL_dcolor * sh_eval(s, d)).sum(), sh_coeffs, self.eps_diff
        )

        print("sh_grad (deg 0+1+2)")
        print(f"  sh_coeffs  rel err = {rel_err(dL_d_sh_a, dL_d_sh_n):.2e}")

        self.assertLess(
            rel_err(dL_d_sh_a, dL_d_sh_n),
            self.eps_thrs,
            f"sh_coeffs rel err {rel_err(dL_d_sh_a, dL_d_sh_n):.2e} >= {self.eps_thrs}",
        )

    def test_grad_colored_rendering(self):
        rng = np.random.default_rng(42)

        xs = np.linspace(-3, 3, 20)
        ys = np.linspace(-3, 3, 20)
        grid_x, grid_y = np.meshgrid(xs, ys)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        scale = np.array([1.5, 1.0, 0.7])
        q = Rotation.random(random_state=42).as_quat()
        pos = np.array([0.1, -0.2, 0.0])
        view_matrix = np.eye(4)

        cov_2d, mu_2d = projected_gaussian_2d(
            pos, scale, quat_to_matrix(q), view_matrix
        )
        G2, _ = eval_gaussian_2d(np.linalg.inv(cov_2d), mu_2d, grid_x, grid_y)

        color1 = rng.uniform(0.2, 0.8, 3)
        G1 = rng.uniform(0, 1, G2.shape)
        sh_coeffs = rng.standard_normal((1, 3))
        d = np.zeros(3)  # direction irrelevant for l=0

        def loss(s):
            color2 = sh_eval(s, d)
            colored_diff = G2[:, :, None] * color2 - G1[:, :, None] * color1
            return 0.5 * (colored_diff**2).sum() * dx * dy

        color2 = sh_eval(sh_coeffs, d)
        colored_diff = G2[:, :, None] * color2 - G1[:, :, None] * color1
        dL_d_rendered = colored_diff * dx * dy
        dL_d_color2 = (dL_d_rendered * G2[:, :, None]).sum(axis=(0, 1))
        dL_d_sh_a = sh_grad(sh_coeffs, d, dL_d_color2)
        dL_d_sh_n = finite_diff_grad(loss, sh_coeffs, self.eps_diff)

        print("grad_colored_rendering")
        print(f"  sh_coeffs  rel err = {rel_err(dL_d_sh_a, dL_d_sh_n):.2e}")

        self.assertLess(
            rel_err(dL_d_sh_a, dL_d_sh_n),
            self.eps_thrs,
            f"sh_coeffs rel err {rel_err(dL_d_sh_a, dL_d_sh_n):.2e} >= {self.eps_thrs}",
        )


if __name__ == "__main__":
    unittest.main()
