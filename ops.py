import numpy as np
from forward import projected_gaussian_2d, eval_gaussian_2d
from backward import grad_eval_gaussian_2d, grad_projected_gaussian_2d
from kernels import sh_eval, sh_grad, quat_to_matrix


class SHEvalOp:
    def forward(self, ctx):
        n = ctx["n_active"]
        ctx["color"] = sh_eval(ctx["sh_coeffs"][:n], ctx["view_dir"])

    def backward(self, ctx):
        n = ctx["n_active"]
        dL_dsh = np.zeros_like(ctx["sh_coeffs"])
        dL_dsh[:n] = sh_grad(ctx["sh_coeffs"][:n], ctx["view_dir"], ctx["color_dL"])
        ctx["sh_coeffs_dL"] = dL_dsh


class ProjectGaussianOp:
    def forward(self, ctx):
        cov, mu = projected_gaussian_2d(
            ctx["pos"], ctx["scale"], quat_to_matrix(ctx["q"]), ctx["view"]
        )
        ctx["precision"] = np.linalg.inv(cov)
        ctx["mu"] = mu

    def backward(self, ctx):
        dL_dpos, dL_dscale, dL_dq = grad_projected_gaussian_2d(
            ctx["precision"], ctx["scale"], ctx["q"], ctx["view"],
            ctx["precision_dL"], ctx["mu_dL"],
        )
        ctx["pos_dL"] = dL_dpos
        ctx["scale_dL"] = dL_dscale
        ctx["q_dL"] = dL_dq


class RasterizeGaussianOp:
    def forward(self, ctx):
        G, cache = eval_gaussian_2d(
            ctx["precision"], ctx["mu"], ctx["grid_x"], ctx["grid_y"]
        )
        ctx["G"] = G
        ctx["_cache_G"] = cache

    def backward(self, ctx):
        dL_dP, dL_dmu = grad_eval_gaussian_2d(
            ctx["_cache_G"], ctx["precision"], ctx["G_dL"]
        )
        ctx["precision_dL"] = dL_dP
        ctx["mu_dL"] = dL_dmu


class ColoredRenderLossOp:
    """MSE loss over the colored render; l1 stored as a display metric only."""

    def forward(self, ctx):
        diff = ctx["G"][:, :, None] * ctx["color"] - ctx["G1"][:, :, None] * ctx["color1"]
        ctx["colored_diff"] = diff
        ctx["l1"] = np.abs(diff).sum() * ctx["dx"] * ctx["dy"]

    def backward(self, ctx):
        dL_d_rendered = ctx["colored_diff"] * ctx["dx"] * ctx["dy"]
        ctx["color_dL"] = (dL_d_rendered * ctx["G"][:, :, None]).sum(axis=(0, 1))
        ctx["G_dL"] = (dL_d_rendered * ctx["color"]).sum(axis=2)


_OPS = [SHEvalOp(), ProjectGaussianOp(), RasterizeGaussianOp(), ColoredRenderLossOp()]


def run_forward(ctx):
    stack = []
    for op in _OPS:
        op.forward(ctx)
        stack.append(op)
    return stack


def run_backward(ctx, stack):
    while stack:
        stack.pop().backward(ctx)
