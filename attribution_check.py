import unittest
import numpy as np
from ops import run_forward, run_backward
from attribution import compute_attribution_maps


class TestAttribution(unittest.TestCase):
    tol = 1e-10

    def setUp(self):
        rng = np.random.default_rng(0)
        grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
        dx = grid_x[0, 1] - grid_x[0, 0]
        dy = grid_y[1, 0] - grid_y[0, 0]
        view = np.eye(4)

        ctx = {
            "sh_coeffs": rng.uniform(-0.5, 0.5, (9, 3)),
            "n_active": 9,
            "view_dir": view[2, :3],
            "position": rng.standard_normal(3) * 0.3,
            "scale": np.abs(rng.standard_normal(3)) * 0.5 + 0.3,
            "rotation": rng.standard_normal(4),
            "view": view,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "G1": np.zeros((60, 60)),
            "color1": np.zeros(3),
            "dx": dx,
            "dy": dy,
        }
        ctx["rotation"] /= np.linalg.norm(ctx["rotation"])

        stack = run_forward(ctx)
        run_backward(ctx, stack)
        self.maps = compute_attribution_maps(ctx)
        self.ctx = ctx

    def test_pos_gradient_reconstruction(self):
        ctx = self.ctx
        d0, d1, G = ctx["_cache_G"]
        precision = ctx["precision"]
        G_dL = ctx["G_dL"]
        view = ctx["view"]

        Pq0 = precision[0, 0] * d0 + precision[0, 1] * d1
        Pq1 = precision[1, 0] * d0 + precision[1, 1] * d1
        w = G_dL * G
        dmu = w[:, :, None] * np.stack([Pq0, Pq1], axis=-1)
        R_v = view[:3, :3]
        pos_reconstructed = (dmu @ R_v[:2, :]).sum(axis=(0, 1))

        err = np.abs(pos_reconstructed - ctx["position_dL"]).max()
        print(f"pos reconstruction error: {err:.2e}")
        self.assertLess(
            err, self.tol, f"pos reconstruction error {err:.2e} >= {self.tol}"
        )

    def test_attribution_maps_nonnegative(self):
        for key, arr in self.maps.items():
            with self.subTest(key=key):
                print(f"{key:10s}  max={arr.max():.4f}  mean={arr.mean():.4f}")
                self.assertTrue((arr >= 0).all(), f"{key} has negative values")


if __name__ == "__main__":
    unittest.main()
