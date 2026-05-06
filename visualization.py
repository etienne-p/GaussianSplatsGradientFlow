import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from forward import ortho_proj_ellipse_matrix
from kernels import sh_eval

_PANE_COLOR = "#4b4b4b"

plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": _PANE_COLOR,
    "axes.labelcolor": "white",
    "axes.labelsize": 9,
    "axes.titlecolor": "white",
    "axes.titlesize": 9,
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def create_ellipsoid(position, scale, rotation, res=40):
    """
    Generate ellipsoid surface data.

    PARAMETERS
    position : (3,), center position (x, y, z)
    scale    : (3,), semi-axes lengths (a, b, c) along X, Y, Z
    rotation : (3, 3), rotation matrix applied before translation
    res      : int, resolution of the parametric surface

    RETURNS
    Xs, Ys, Zs : surface coordinates for plotting
    """
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res // 2)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    a, b, c = scale
    pts = np.stack([a * xs.ravel(), b * ys.ravel(), c * zs.ravel()])
    pts_rot = rotation @ pts
    pts_final = pts_rot + position[:, np.newaxis]

    return (
        pts_final[0].reshape(xs.shape),
        pts_final[1].reshape(ys.shape),
        pts_final[2].reshape(zs.shape),
    )


def draw_3d_projection(ax, view_matrix, A_proj, depth_offset, alpha, proj_color):
    """
    Draw a projected ellipse back into 3D world space.

    PARAMETERS
    view_matrix  : (4, 4), world-to-view transform used for the projection
    A_proj       : (2, 2), precision matrix defining the ellipse  q^T A q = 1
    depth_offset : float, depth along view-space z where the shadow sits
    alpha        : float, fill opacity
    proj_color   : str, fill color
    """
    eigvals, eigvecs = np.linalg.eigh(A_proj)
    scales_2d = 1.0 / np.sqrt(np.clip(eigvals, 1e-12, None))
    theta = np.linspace(0, 2 * np.pi, 200)
    unit = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = eigvecs @ (scales_2d[:, np.newaxis] * unit)

    R_v = view_matrix[:3, :3]
    t_v = view_matrix[:3, 3]
    pts_view = np.vstack([ellipse, np.full((1, ellipse.shape[1]), depth_offset)])
    pts_world = R_v.T @ (pts_view - t_v[:, np.newaxis])
    x3, y3, z3 = pts_world

    verts = [list(zip(x3, y3, z3))]
    poly = Poly3DCollection(
        verts, facecolor=proj_color, alpha=alpha, edgecolor=proj_color, linewidth=2
    )
    ax.add_collection3d(poly)


def draw_gaussian(ax, pos, scale, rotation, sh_coeffs, views, plane_offset, alpha=0.5):
    """
    Draw a single ellipsoid and its orthographic shadows on the projection planes,
    with per-vertex SH colors and degree-0 shadows.

    PARAMETERS
    ax           : Axes3D
    pos          : (3,), center position
    scale        : (3,), semi-axes lengths
    rotation     : (3, 3), rotation matrix
    sh_coeffs    : (n, 3), SH coefficients in sh_lm_index order
    views        : list of (4, 4), one view matrix per shadow plane
    plane_offset : float, distance at which shadow planes sit along view-space z
    alpha        : float, surface opacity; shadows use half this value
    """
    res = 40
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res // 2)
    unit_dirs = np.stack([
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
    ], axis=-1)  # (res, res//2, 3) unit sphere normals in object space
    world_dirs = unit_dirs @ rotation.T   # (res, res//2, 3) rotated to world space
    colors = np.clip(sh_eval(sh_coeffs, world_dirs), 0.0, 1.0)  # clip for display only
    rgba = np.concatenate(
        [colors, np.full(colors.shape[:2] + (1,), alpha)], axis=-1
    )  # (res, res//2, 4)

    Xs, Ys, Zs = create_ellipsoid(pos, scale, rotation, res=res)
    ax.plot_surface(
        Xs, Ys, Zs,
        facecolors=rgba,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False,
    )

    for view in views:
        view_dir = view[2, :3]  # orthographic: constant viewing direction for all points
        shadow_color = np.clip(sh_eval(sh_coeffs, view_dir), 0.0, 1.0)
        cov = ortho_proj_ellipse_matrix(scale, rotation, view)
        draw_3d_projection(ax, view, np.linalg.inv(cov), -plane_offset,
                           alpha=alpha * 0.5, proj_color=shadow_color)


_BLUE_BLACK_RED = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_black_red", ["blue", "black", "red"]
)

_VIEW_LABELS = ["XY", "XZ", "YZ"]


def draw_l1_comparison(ax, diff, xs, ys, l1, label, vmax):
    """Render one view's G2 − G1 signed diff as a diverging heatmap. Returns the image."""
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    im = ax.imshow(
        diff, origin="lower", extent=extent, cmap=_BLUE_BLACK_RED, vmin=-vmax, vmax=vmax
    )
    ax.set_title(f"{label}  L1={l1:.3f}")
    return im


def draw_frame(
    ax3d, ax_diffs, cax,
    diffs, l1s, step, vmax,
    pos1, scale1, R1, sh_coeffs_1,
    pos2, scale2, R2, sh_coeffs_2,
    views, plane_offset,
    xs, ys,
    show_ref=True,
):
    ax3d.cla()
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(_PANE_COLOR)
    ax3d.yaxis.pane.set_edgecolor(_PANE_COLOR)
    ax3d.zaxis.pane.set_edgecolor(_PANE_COLOR)

    if show_ref:
        draw_gaussian(ax3d, pos1, scale1, R1, sh_coeffs_1,
                      views=views, plane_offset=plane_offset)
    draw_gaussian(ax3d, pos2, scale2, R2, sh_coeffs_2,
                  views=views, plane_offset=plane_offset)

    ax3d.set_xlim(-plane_offset, plane_offset)
    ax3d.set_ylim(-plane_offset, plane_offset)
    ax3d.set_zlim(-plane_offset, plane_offset)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"Step {step}", fontsize=10, pad=8)
    ax3d.view_init(elev=25, azim=40)

    im = None
    for ax, diff, l1, label in zip(ax_diffs, diffs, l1s, _VIEW_LABELS):
        ax.cla()
        im = draw_l1_comparison(ax, diff, xs, ys, l1, label, vmax)
    cax.cla()
    plt.colorbar(im, cax=cax)
