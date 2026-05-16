import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons
from forward import eval_gaussian_2d, projected_gaussian_2d
from kernels import quat_to_matrix, sh_eval
from ops import run_forward, run_backward
from attribution import compute_attribution_maps
from optimizer import AdamOptimizer
from visualization import draw_frame

_MAX_FRAMES = 128
_PLANE_OFFSET = 3.5
_BLUE_BLACK_RED = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_black_red", ["blue", "black", "red"]
)
_VIEW_LABELS = ["XY", "XZ", "YZ"]


@dataclass
class _RenderCache:
    """Transient rendering state: persistent artists and cached forward results."""

    attr_maps: object = None
    attr_vmax: dict = field(default_factory=dict)
    im_handles: list = field(default_factory=lambda: [None, None, None])
    cbar: object = None
    loss_line: object = None
    last_diffs: object = None
    last_l1s: object = None


class GaussianOptimizer:
    def __init__(self, lr=0.1, perturbation=0.4):
        self.opt = AdamOptimizer(lr=lr)
        self.perturbation = perturbation

        # Grid (fixed; generous extent for any valid scale1)
        self.xs = np.linspace(-_PLANE_OFFSET, _PLANE_OFFSET, 200)
        self.ys = np.linspace(-_PLANE_OFFSET, _PLANE_OFFSET, 200)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        self.plane_offset = _PLANE_OFFSET

        # Views (fixed)
        view_xy = np.eye(4)
        view_xz = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        view_yz = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        self.views = [view_xy, view_xz, view_yz]

        # Figure and controls
        self.fig, self.ax3d, self.ax_diffs, self.cax, self.ax_loss = (
            self._build_figure()
        )
        self.animation = None
        self.show_ref = True
        self.display_mode = "L1"
        self._cache = _RenderCache()
        self._build_controls()

        # Random initial pair + draw
        self._randomize()
        self._init_images()
        init_diffs, init_l1s = self._forward_diffs()
        self._cache.last_diffs = init_diffs
        self._cache.last_l1s = init_l1s
        self.vmax = max(np.abs(d).max() for d in init_diffs)
        self.l1_history = [sum(init_l1s)]
        self._draw(init_diffs, init_l1s)
        self._start_animation()

    def _randomize(self):
        # Random target: descending scale axes, uniform random rotation
        self.pos1 = np.zeros(3)
        self.scale1 = np.sort(np.random.uniform(0.5, 2.5, 3))[::-1].copy()
        self.R1 = Rotation.random().as_matrix()

        # Perturbed initial: scale perturbed per-axis, rotation offset by fixed angle
        self.pos2 = np.zeros(3)
        scale_factors = np.random.uniform(
            1 - self.perturbation, 1 + self.perturbation, 3
        )
        self.scale2 = np.maximum(self.scale1 * scale_factors, 0.1)

        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = self.perturbation * np.pi / 2
        self.q2 = (
            Rotation.from_rotvec(axis * angle) * Rotation.from_matrix(self.R1)
        ).as_quat()

        # SH coefficients (9, 3): row 0 → l=0, rows 1-3 → l=1, rows 4-8 → l=2
        sh0 = np.random.uniform(-1.5, 1.5, size=(1, 3))
        sh1 = np.random.uniform(-0.8, 0.8, size=(3, 3))
        sh2 = np.random.uniform(-0.3, 0.3, size=(5, 3))
        self.sh_coeffs_1 = np.vstack([sh0, sh1, sh2])
        sh0_p = np.random.uniform(-1.5, 1.5, size=(1, 3))
        # New SH bands are initialized to zero on the candidate,
        # so unlocking a band is a neutral operation:
        # the candidate's rendered color is unchanged at the moment of unlock,
        # and the target is always evaluated at full degree.
        sh1_p = np.zeros((3, 3))
        sh2_p = np.zeros((5, 3))
        self.sh_coeffs_2 = np.vstack([sh0_p, sh1_p, sh2_p])

        # Precompute target projections
        self.G1_views = []
        for view in self.views:
            cov1, mu1 = projected_gaussian_2d(self.pos1, self.scale1, self.R1, view)
            G1, _ = eval_gaussian_2d(np.linalg.inv(cov1), mu1, self.grid_x, self.grid_y)
            self.G1_views.append(G1)

        self.opt.reset()
        self.step_counter = 0

    def _build_figure(self):
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[1.1, 1], wspace=0.15, bottom=0.13, top=0.97
        )
        ax3d = fig.add_subplot(gs[0], projection="3d")
        gs_right = gs[1].subgridspec(1, 2, width_ratios=[1, 0.06], wspace=0.08)
        gs_plots = gs_right[0].subgridspec(2, 2, hspace=0.5, wspace=0.35)
        ax_diffs = [
            fig.add_subplot(gs_plots[0, 0]),
            fig.add_subplot(gs_plots[0, 1]),
            fig.add_subplot(gs_plots[1, 0]),
        ]
        ax_loss = fig.add_subplot(gs_plots[1, 1])
        cax = fig.add_subplot(gs_right[1])
        fig.text(
            0.5,
            0.99,
            "3DGS - Gradient Backpropagation",
            ha="center",
            va="top",
            fontsize=12,
        )
        return fig, ax3d, ax_diffs, cax, ax_loss

    def _build_controls(self):
        btn_w, btn_h = 0.1, 0.05
        btn_y = 0.03
        centers = [0.25, 0.36, 0.47, 0.58, 0.72]
        labels = ["Reset", "Pause", "Next", "Resume", "Hide Ref"]
        btn_axes = [
            self.fig.add_axes([cx - btn_w / 2, btn_y, btn_w, btn_h]) for cx in centers
        ]
        self.btn_reset, self.btn_pause, self.btn_next, self.btn_resume, self.btn_ref = [
            Button(ax, lbl, color="#333333", hovercolor="#555555")
            for ax, lbl in zip(btn_axes, labels)
        ]
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_next.on_clicked(self._on_next)
        self.btn_resume.on_clicked(self._on_resume)
        self.btn_ref.on_clicked(self._on_toggle_ref)

        ax_radio = self.fig.add_axes([0.005, 0.005, 0.2, 0.2])
        ax_radio.set_aspect("equal")
        self.radio = RadioButtons(
            ax_radio,
            ["L1", "position", "scale", "rotation", "sh_coeffs"],
            activecolor="white",
        )
        labels_text = [lbl.get_text() for lbl in self.radio.labels]
        for circle, text in zip(self.radio.circles, labels_text):
            circle.set_edgecolor("white")
            circle.set_facecolor("white" if text == self.display_mode else "black")
        self.radio.on_clicked(self._on_mode_change)

    def _init_images(self):
        extent = [self.xs[0], self.xs[-1], self.ys[0], self.ys[-1]]
        blank = np.zeros((len(self.ys), len(self.xs)))
        for i, ax in enumerate(self.ax_diffs):
            self._cache.im_handles[i] = ax.imshow(
                blank,
                origin="lower",
                extent=extent,
                cmap=_BLUE_BLACK_RED,
                vmin=-1,
                vmax=1,
            )
        self._cache.cbar = self.fig.colorbar(self._cache.im_handles[-1], cax=self.cax)
        (self._cache.loss_line,) = self.ax_loss.plot([], [], color="white", linewidth=1)
        self.ax_loss.set_title("L1 error")

    def _start_animation(self):
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=_MAX_FRAMES,
            interval=200,
            repeat=False,
            cache_frame_data=False,
        )

    def _on_reset(self, _event):
        self.animation.event_source.stop()
        self._randomize()
        self._cache.attr_maps = None
        self._cache.attr_vmax = {}
        init_diffs, init_l1s = self._forward_diffs()
        self._cache.last_diffs = init_diffs
        self._cache.last_l1s = init_l1s
        self.vmax = max(np.abs(d).max() for d in init_diffs)
        self.l1_history = [sum(init_l1s)]
        self._draw(init_diffs, init_l1s)
        self.fig.canvas.draw_idle()
        self._start_animation()

    def _on_pause(self, _event):
        self.animation.event_source.stop()

    def _on_next(self, _event):
        self.animation.event_source.stop()
        if self.step_counter < _MAX_FRAMES:
            self.update(None)
            self.fig.canvas.draw_idle()

    def _on_resume(self, _event):
        if self.step_counter >= _MAX_FRAMES:
            return
        self.animation.event_source.start()

    def _on_mode_change(self, label):
        self.display_mode = label
        self._draw(self._cache.last_diffs, self._cache.last_l1s)
        self.fig.canvas.draw_idle()

    def _on_toggle_ref(self, _event):
        self.show_ref = not self.show_ref
        self.btn_ref.label.set_text("Hide Ref" if self.show_ref else "Show Ref")
        self._draw(self._cache.last_diffs, self._cache.last_l1s)
        self.fig.canvas.draw_idle()

    def _active_sh_count(self):
        if self.step_counter < _MAX_FRAMES // 3:
            return 1
        if self.step_counter < 2 * (_MAX_FRAMES // 3):
            return 4
        return 9

    def _build_ctx(self, view, G1_v, n):
        view_dir = view[2, :3]
        return {
            "sh_coeffs": self.sh_coeffs_2,
            "n_active": n,
            "view_dir": view_dir,
            "position": self.pos2,
            "scale": self.scale2,
            "rotation": self.q2,
            "view": view,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "G1": G1_v,
            "color1": sh_eval(self.sh_coeffs_1, view_dir),
            "dx": self.dx,
            "dy": self.dy,
        }

    def _forward_diffs(self):
        n = self._active_sh_count()
        diffs, l1s = [], []
        for view, G1_v in zip(self.views, self.G1_views):
            ctx = self._build_ctx(view, G1_v, n)
            run_forward(ctx)
            diffs.append(ctx["colored_diff"].mean(axis=2))
            l1s.append(ctx["l1"])
        return diffs, l1s

    def _draw(self, diffs, l1s):
        draw_frame(
            self.ax3d,
            self.step_counter,
            self.pos1,
            self.scale1,
            self.R1,
            self.sh_coeffs_1,
            self.pos2,
            self.scale2,
            quat_to_matrix(self.q2),
            self.sh_coeffs_2,
            self.views,
            self.plane_offset,
            show_ref=self.show_ref,
        )

        if self.display_mode == "L1":
            cmap, vmin, vmax_2d = _BLUE_BLACK_RED, -self.vmax, self.vmax
            items = [
                (diff, f"{lbl}  L1={l1:.3f}")
                for diff, l1, lbl in zip(diffs, l1s, _VIEW_LABELS)
            ]
        elif self._cache.attr_vmax:
            key = self.display_mode
            cmap, vmin, vmax_2d = "hot", 0.0, self._cache.attr_vmax[key]
            items = [
                (m[key], lbl) for m, lbl in zip(self._cache.attr_maps, _VIEW_LABELS)
            ]
        else:
            for ax, lbl in zip(self.ax_diffs, _VIEW_LABELS):
                ax.set_title(lbl)
            self._update_loss_line()
            return

        for i, (ax, (data, title)) in enumerate(zip(self.ax_diffs, items)):
            self._cache.im_handles[i].set_data(data)
            self._cache.im_handles[i].set_clim(vmin=vmin, vmax=vmax_2d)
            self._cache.im_handles[i].set_cmap(cmap)
            ax.set_title(title)

        self._cache.cbar.update_normal(self._cache.im_handles[-1])

        self._update_loss_line()

    def _update_loss_line(self):
        self._cache.loss_line.set_data(range(len(self.l1_history)), self.l1_history)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

    def step(self, view, G1_v):
        n = self._active_sh_count()
        ctx = self._build_ctx(view, G1_v, n)
        stack = run_forward(ctx)
        run_backward(ctx, stack)

        self.pos2 = self.opt.step("pos", self.pos2, ctx["position_dL"])
        self.scale2 = np.maximum(
            self.opt.step("scale", self.scale2, ctx["scale_dL"]), 1e-3
        )
        self.q2 = self.opt.step("q", self.q2, ctx["rotation_dL"])
        self.q2 /= np.linalg.norm(self.q2)
        self.sh_coeffs_2 = self.opt.step(
            "sh_coeffs", self.sh_coeffs_2, ctx["sh_coeffs_dL"]
        )

        return (
            ctx["colored_diff"].mean(axis=2),
            ctx["l1"],
            compute_attribution_maps(ctx),
        )

    def update(self, _frame):
        if self.step_counter >= _MAX_FRAMES:
            self.animation.event_source.stop()
            return
        results = [
            self.step(view, G1_v) for view, G1_v in zip(self.views, self.G1_views)
        ]
        diffs = [r[0] for r in results]
        l1s = [r[1] for r in results]
        self._cache.attr_maps = [r[2] for r in results]
        if not self._cache.attr_vmax:
            for key in self._cache.attr_maps[0]:
                self._cache.attr_vmax[key] = max(
                    max(m[key].max() for m in self._cache.attr_maps), 1e-12
                )
        self._cache.last_diffs = diffs
        self._cache.last_l1s = l1s
        self.step_counter += 1
        if self.step_counter == _MAX_FRAMES // 3:
            print(f"Step {self.step_counter}: unlocking SH degree 1")
        elif self.step_counter == 2 * (_MAX_FRAMES // 3):
            print(f"Step {self.step_counter}: unlocking SH degree 2")
        self.l1_history.append(sum(l1s))
        self._draw(diffs, l1s)

    def run(self):
        plt.show()


if __name__ == "__main__":
    GaussianOptimizer().run()
