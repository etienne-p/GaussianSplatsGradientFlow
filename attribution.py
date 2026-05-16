import numpy as np
from backward import grad_eval_gaussian_2d_per_pixel, grad_projected_gaussian_2d


def compute_attribution_maps(ctx):
    """
    Per-pixel gradient attribution maps for each splat parameter group.

    For each pixel (i,j) and parameter group p, answers: how much does the
    current per-pixel error at (i,j) contribute to p's gradient update?

    Requires ctx populated by run_forward (colored_diff, G, _cache_G,
    precision, color, rotation, scale, view, dx, dy).

    We use the norm of the per-pixel contributions as the attribution value,
    so that we obtain 2D visualizations.

    RETURNS
    dict mapping 'position' | 'scale' | 'rotation' | 'sh_coeffs' to (H, W) arrays,
    each the L2 norm of that parameter group's gradient contribution per pixel.
    """
    diff = ctx["colored_diff"]  # (H, W, 3)
    G = ctx["_cache_G"][2]  # (H, W)
    color = ctx["color"]  # (3,)
    dx, dy = ctx["dx"], ctx["dy"]

    dL_drendered = diff * dx * dy  # (H, W, 3)
    G_dL = (dL_drendered * color).sum(axis=-1)  # (H, W)

    # SH / color attribution
    # Per-pixel contribution to color_dL = dL_drendered[ij] * G[ij]
    sh_attr = np.linalg.norm(dL_drendered * G[:, :, None], axis=-1)  # (H, W)

    # Geometry attribution
    # grad_eval_gaussian_2d_per_pixel returns (H,W,2,2) and (H,W,2) — the
    # per-pixel contributions before the spatial sum.
    dP, dmu = grad_eval_gaussian_2d_per_pixel(
        ctx["_cache_G"], ctx["precision"], G_dL
    )  # (H,W,2,2), (H,W,2)

    # Batched gradient evaluation.
    pos_pp, scale_pp, q_pp = grad_projected_gaussian_2d(
        ctx["precision"], ctx["scale"], ctx["rotation"], ctx["view"], dP, dmu
    )  # (H,W,3), (H,W,3), (H,W,4)

    return {
        "position": np.linalg.norm(pos_pp, axis=-1),
        "scale": np.linalg.norm(scale_pp, axis=-1),
        "rotation": np.linalg.norm(q_pp, axis=-1),
        "sh_coeffs": sh_attr,
    }
