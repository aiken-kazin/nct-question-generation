"""
Textbook-style Matplotlib figures for exam questions.

Visual rules (mirror Kazakhstani school textbooks):
  - Black and white only — no color fills, no colored lines
  - No gridlines
  - Geometric figures float on white canvas (no axes box)
  - Function graphs use clean axes with arrowheads through the origin
  - Measurements (side lengths, angles) labeled directly on the figure
  - Construction lines (altitudes, radii to chords) are dashed
  - Hatching (not color) for shaded areas
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from .models import FigureSpec

# ── Safe expression evaluator ────────────────────────────────────────────────

_SAFE_NS: dict = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs,
    "pi": np.pi, "e": np.e,
    "__builtins__": {},
}


def _eval_func(expr: str, x: np.ndarray) -> np.ndarray:
    ns = {**_SAFE_NS, "x": x}
    return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307


# ── Shared style constants ────────────────────────────────────────────────────

_LW = 1.5       # main line width
_LW_THIN = 0.8  # construction / secondary lines
_FS_LBL = 11    # vertex / point label font size
_FS_MSR = 9     # measurement label font size
_DOT = 4        # marker size for key points


def _geometry_fig(figsize=(5, 5)):
    """White canvas, no axes — for geometric figures."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _graph_fig(figsize=(6, 5)):
    """White background, spines through origin, no box — for function graphs."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", which="both", direction="in",
                   width=0.6, length=4, labelsize=8, color="black")
    ax.grid(False)
    return fig, ax


def _add_axis_arrows(ax):
    """Extend axes with arrowheads and x / y labels at the tips."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    dx = (xr - xl) * 0.05
    dy = (yt - yb) * 0.05

    kw = dict(arrowstyle="->", color="black", lw=0.9)
    ax.annotate("", xy=(xr + dx, 0), xytext=(xr + dx * 0.01, 0),
                arrowprops=kw, annotation_clip=False)
    ax.annotate("", xy=(0, yt + dy), xytext=(0, yt + dy * 0.01),
                arrowprops=kw, annotation_clip=False)
    ax.text(xr + dx * 1.8, 0, "x", ha="left", va="center", fontsize=10)
    ax.text(0, yt + dy * 2, "y", ha="center", va="bottom", fontsize=10)


# ── FigureGenerator ───────────────────────────────────────────────────────────

class FigureGenerator:
    def generate(self, spec: FigureSpec, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig_id = f"{spec.figure_type}_{abs(hash(str(spec.parameters))) % 100000:05d}"
        out_path = output_dir / f"{fig_id}.png"

        handlers = {
            "triangle": self._triangle,
            "circle": self._circle,
            "function_graph": self._function_graph,
            "trig_graph": self._trig_graph,
            "vector_diagram": self._vector_diagram,
            "sequence_plot": self._sequence_plot,
            "solid_3d": self._solid_3d,
            "coordinate_plane": self._coordinate_plane,
        }

        handler = handlers.get(spec.figure_type, self._placeholder)
        fig, ax = handler(spec.parameters)

        if spec.caption:
            fig.text(0.5, 0.01, spec.caption, ha="center", va="bottom",
                     fontsize=9, style="italic", color="black")

        fig.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return out_path

    # ── Triangle ─────────────────────────────────────────────────────────────

    def _triangle(self, p: dict):
        verts_raw = p.get("vertices", [[0, 0], [6, 0], [2.5, 5]])
        labels = p.get("labels", ["A", "B", "C"])
        side_lengths: dict = p.get("side_lengths", {}) or {}
        angles_deg: dict = p.get("angles", {}) or {}
        right_v = p.get("right_angle_vertex")
        show_height = p.get("show_height")
        height_foot_label = p.get("height_foot_label", "H")
        show_incircle = p.get("show_incircle", False)
        incircle_r_label = p.get("incircle_radius_label")
        show_circumcircle = p.get("show_circumcircle", False)

        v = np.array(verts_raw, dtype=float)
        fig, ax = _geometry_fig()

        # Draw sides
        tri = plt.Polygon(v, fill=False, edgecolor="black", linewidth=_LW, zorder=2)
        ax.add_patch(tri)

        # Vertex labels — offset outward from centroid
        centroid = v.mean(axis=0)
        for lbl, pt in zip(labels, v):
            direction = pt - centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * 0.4
            ax.text(pt[0] + direction[0], pt[1] + direction[1],
                    lbl, fontsize=_FS_LBL, fontweight="bold",
                    ha="center", va="center", zorder=5)

        # Side length labels
        label_chars = list("".join(labels))
        side_pairs = [
            (0, 1, labels[0] + labels[1]),
            (1, 2, labels[1] + labels[2]),
            (2, 0, labels[2] + labels[0]),
        ]
        for i, j, key in side_pairs:
            val = side_lengths.get(key)
            if val is None:
                # Try reversed key
                val = side_lengths.get(key[::-1])
            if val is not None:
                mid = (v[i] + v[j]) / 2
                perp = _perp_offset(v[i], v[j], 0.22)
                ax.text(mid[0] + perp[0], mid[1] + perp[1],
                        str(val), fontsize=_FS_MSR, ha="center", va="center",
                        zorder=4)

        # Angle arcs and labels
        for lbl, pt, (prev_idx, next_idx) in zip(
            labels, v, [(2, 1), (0, 2), (1, 0)]
        ):
            deg_val = angles_deg.get(lbl)
            if lbl == right_v:
                _draw_right_angle_marker(ax, v, list(labels).index(lbl))
            elif deg_val is not None:
                _draw_angle_arc(ax, pt, v[prev_idx], v[next_idx], str(deg_val) + "°")

        # Altitude (height)
        if show_height and show_height in labels:
            idx = labels.index(show_height)
            opp_a = v[(idx + 1) % 3]
            opp_b = v[(idx + 2) % 3]
            foot = _foot_of_perpendicular(v[idx], opp_a, opp_b)
            ax.plot([v[idx][0], foot[0]], [v[idx][1], foot[1]],
                    "k--", linewidth=_LW_THIN, zorder=1)
            ax.plot(*foot, "ko", markersize=3, zorder=3)
            if height_foot_label:
                offs = _perp_offset(opp_a, opp_b, 0.22)
                ax.text(foot[0] + offs[0], foot[1] - 0.28,
                        height_foot_label, fontsize=_FS_MSR - 1,
                        ha="center", va="top", zorder=4)
            # Right angle at foot
            _draw_right_angle_at(ax, foot, v[idx], opp_b, size=0.18)

        # Inscribed circle
        if show_incircle:
            ic, ir = _incircle(v)
            circ = plt.Circle(ic, ir, fill=False, edgecolor="black",
                              linewidth=_LW_THIN, linestyle="--", zorder=1)
            ax.add_patch(circ)
            ax.plot(*ic, "k.", markersize=3, zorder=3)
            if incircle_r_label:
                # Draw radius to closest side
                ax.annotate("", xy=(ic[0], ic[1] - ir), xytext=ic,
                            arrowprops=dict(arrowstyle="-", color="black",
                                            lw=_LW_THIN))
                ax.text(ic[0] + 0.1, ic[1] - ir / 2, incircle_r_label,
                        fontsize=_FS_MSR, ha="left", zorder=4)

        # Circumscribed circle
        if show_circumcircle:
            cc, cr = _circumcircle(v)
            if cc is not None:
                circ = plt.Circle(cc, cr, fill=False, edgecolor="black",
                                  linewidth=_LW_THIN, linestyle="--", zorder=1)
                ax.add_patch(circ)

        _set_triangle_limits(ax, v)
        return fig, ax

    # ── Circle ────────────────────────────────────────────────────────────────

    def _circle(self, p: dict):
        radius = float(p.get("radius", 5))
        r_label = p.get("radius_label")
        chord_eps = p.get("chord_endpoints_deg")
        chord_labels = p.get("chord_labels", ["A", "B"])
        chord_len_label = p.get("chord_length_label")
        inscribed_deg = p.get("inscribed_angle_deg")
        central_label = p.get("central_angle_label")
        tangent_deg = p.get("tangent_at_deg")
        tangent_lbl = p.get("tangent_point_label", "T")

        fig, ax = _geometry_fig()

        # Main circle
        circle = plt.Circle((0, 0), radius, fill=False,
                             edgecolor="black", linewidth=_LW, zorder=2)
        ax.add_patch(circle)

        # Center
        ax.plot(0, 0, "ko", markersize=3, zorder=4)
        ax.text(0.15, 0.15, "O", fontsize=_FS_LBL, fontweight="bold", zorder=5)

        # Radius line + label
        if r_label or radius:
            ax.plot([0, radius], [0, 0], "k-", linewidth=_LW_THIN, zorder=1)
            ax.text(radius / 2, 0.25,
                    r_label if r_label else f"{radius}",
                    fontsize=_FS_MSR, ha="center", zorder=4)

        pt_A = pt_B = None
        if chord_eps and len(chord_eps) == 2:
            a1 = math.radians(chord_eps[0])
            a2 = math.radians(chord_eps[1])
            pt_A = np.array([radius * math.cos(a1), radius * math.sin(a1)])
            pt_B = np.array([radius * math.cos(a2), radius * math.sin(a2)])

            # Chord
            ax.plot([pt_A[0], pt_B[0]], [pt_A[1], pt_B[1]],
                    "k-", linewidth=_LW, zorder=2)

            # Dashed radii to chord endpoints
            ax.plot([0, pt_A[0]], [0, pt_A[1]], "k--", linewidth=_LW_THIN, zorder=1)
            ax.plot([0, pt_B[0]], [0, pt_B[1]], "k--", linewidth=_LW_THIN, zorder=1)

            # Endpoint labels
            for lbl, pt in zip(chord_labels[:2], [pt_A, pt_B]):
                off = pt / np.linalg.norm(pt) * 0.35
                ax.text(pt[0] + off[0], pt[1] + off[1], lbl,
                        fontsize=_FS_LBL, fontweight="bold", ha="center", zorder=5)
                ax.plot(*pt, "ko", markersize=3, zorder=4)

            # Chord length label
            if chord_len_label:
                mid = (pt_A + pt_B) / 2
                perp = _perp_offset(pt_A, pt_B, 0.28)
                ax.text(mid[0] + perp[0], mid[1] + perp[1], chord_len_label,
                        fontsize=_FS_MSR, ha="center", zorder=4)

            # Central angle arc
            if central_label:
                ang_start = math.degrees(math.atan2(pt_A[1], pt_A[0]))
                ang_end = math.degrees(math.atan2(pt_B[1], pt_B[0]))
                arc_r = radius * 0.25
                arc = mpatches.Arc((0, 0), 2 * arc_r, 2 * arc_r,
                                   angle=0, theta1=min(ang_start, ang_end),
                                   theta2=max(ang_start, ang_end),
                                   color="black", linewidth=_LW_THIN)
                ax.add_patch(arc)
                mid_ang = math.radians((ang_start + ang_end) / 2)
                ax.text(arc_r * 1.15 * math.cos(mid_ang),
                        arc_r * 1.15 * math.sin(mid_ang),
                        central_label, fontsize=_FS_MSR, ha="center", zorder=4)

        # Inscribed angle
        if inscribed_deg is not None and pt_A is not None and pt_B is not None:
            ang_c = math.radians(inscribed_deg)
            pt_C = np.array([radius * math.cos(ang_c), radius * math.sin(ang_c)])
            ax.plot(*pt_C, "ko", markersize=3, zorder=4)
            off_c = pt_C / np.linalg.norm(pt_C) * 0.35
            lbl_c = chord_labels[2] if len(chord_labels) > 2 else "C"
            ax.text(pt_C[0] + off_c[0], pt_C[1] + off_c[1], lbl_c,
                    fontsize=_FS_LBL, fontweight="bold", ha="center", zorder=5)
            ax.plot([pt_C[0], pt_A[0]], [pt_C[1], pt_A[1]],
                    "k-", linewidth=_LW_THIN, zorder=1)
            ax.plot([pt_C[0], pt_B[0]], [pt_C[1], pt_B[1]],
                    "k-", linewidth=_LW_THIN, zorder=1)

        # Tangent
        if tangent_deg is not None:
            ang_t = math.radians(tangent_deg)
            pt_T = np.array([radius * math.cos(ang_t), radius * math.sin(ang_t)])
            tang_dir = np.array([-math.sin(ang_t), math.cos(ang_t)])
            t_len = radius * 0.9
            ax.plot([pt_T[0] - t_len * tang_dir[0], pt_T[0] + t_len * tang_dir[0]],
                    [pt_T[1] - t_len * tang_dir[1], pt_T[1] + t_len * tang_dir[1]],
                    "k-", linewidth=_LW, zorder=2)
            # Radius to tangent point (dashed)
            ax.plot([0, pt_T[0]], [0, pt_T[1]], "k--", linewidth=_LW_THIN, zorder=1)
            ax.plot(*pt_T, "ko", markersize=3, zorder=4)
            off_t = pt_T / np.linalg.norm(pt_T) * 0.35
            ax.text(pt_T[0] + off_t[0], pt_T[1] + off_t[1], tangent_lbl,
                    fontsize=_FS_LBL, fontweight="bold", ha="center", zorder=5)
            # Right angle at tangent point
            _draw_right_angle_at(ax, pt_T,
                                 np.array([0.0, 0.0]),
                                 pt_T + np.array([-math.sin(ang_t), math.cos(ang_t)]),
                                 size=0.22)

        m = radius * 1.55
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        return fig, ax

    # ── Function graph ────────────────────────────────────────────────────────

    def _function_graph(self, p: dict):
        func_str = p.get("function_str", "x**2")
        x_range = p.get("x_range", [-5, 5])
        y_range = p.get("y_range")
        shade = p.get("shade_area")
        mark_pts = p.get("mark_points") or []
        tangent_x = p.get("tangent_at_x")
        func2_str = p.get("second_function_str")
        func_label = p.get("function_label")

        fig, ax = _graph_fig()

        x = np.linspace(x_range[0], x_range[1], 1000)
        try:
            y = _eval_func(func_str, x)
        except Exception:
            y = np.zeros_like(x)

        ax.plot(x, y, "k-", linewidth=_LW, zorder=3)

        if func2_str:
            try:
                y2 = _eval_func(func2_str, x)
                ax.plot(x, y2, "k--", linewidth=_LW_THIN, zorder=2)
            except Exception:
                pass

        # Shaded area — cross-hatch, no color fill
        if shade and len(shade) == 2:
            xa = np.linspace(shade[0], shade[1], 500)
            ya = _eval_func(func_str, xa)
            ax.fill_between(xa, ya, 0, hatch="///", facecolor="none",
                            edgecolor="black", linewidth=0.4, zorder=2)
            # Boundary markers
            for xb in shade:
                yb = float(_eval_func(func_str, np.array([xb]))[0])
                ax.plot([xb, xb], [0, yb], "k--", linewidth=_LW_THIN, zorder=1)

        # Mark key points
        for xm in mark_pts:
            try:
                ym = float(_eval_func(func_str, np.array([float(xm)]))[0])
                ax.plot(xm, ym, "ko", markersize=_DOT, zorder=5)
                # Dashed drop lines
                ax.plot([xm, xm], [0, ym], "k--", linewidth=_LW_THIN - 0.1, zorder=1)
                ax.plot([0, xm], [ym, ym], "k--", linewidth=_LW_THIN - 0.1, zorder=1)
                # Coordinate label
                ax.text(xm + 0.08, ym + 0.12 * (1 if ym >= 0 else -1),
                        f"({_fmt_num(xm)}, {_fmt_num(ym)})",
                        fontsize=7, ha="left", zorder=6)
            except Exception:
                pass

        # Tangent line
        if tangent_x is not None:
            try:
                xt = float(tangent_x)
                dx_eps = 1e-5
                yt = float(_eval_func(func_str, np.array([xt]))[0])
                slope = (float(_eval_func(func_str, np.array([xt + dx_eps]))[0]) - yt) / dx_eps
                tx = np.linspace(x_range[0], x_range[1], 200)
                ty = slope * (tx - xt) + yt
                ax.plot(tx, ty, "k--", linewidth=_LW_THIN, zorder=2)
                ax.plot(xt, yt, "ko", markersize=_DOT, zorder=5)
            except Exception:
                pass

        # Function label near curve end
        if func_label:
            try:
                x_lbl = x_range[1] * 0.9
                y_lbl = float(_eval_func(func_str, np.array([x_lbl]))[0])
                ax.text(x_lbl, y_lbl + 0.2, func_label, fontsize=9, ha="right")
            except Exception:
                pass

        # Finalize axes
        if y_range:
            ax.set_ylim(y_range)
        ax.set_xlim(x_range[0] - 0.3, x_range[1] + 0.5)
        _clean_ticks(ax, x_range, ax.get_ylim())
        _add_axis_arrows(ax)
        return fig, ax

    # ── Trig graph ────────────────────────────────────────────────────────────

    def _trig_graph(self, p: dict):
        func_name = p.get("function", "sin")
        A = float(p.get("amplitude", 1))
        B = float(p.get("frequency_mult", 1))
        C = float(p.get("phase_shift", 0))
        D = float(p.get("vertical_shift", 0))
        n_periods = float(p.get("x_periods", 2))
        mark_xs = p.get("mark_x_values") or []

        period = 2 * np.pi / abs(B) if B != 0 else 2 * np.pi
        x_max = period * n_periods
        x = np.linspace(-0.1, x_max + 0.2, 2000)

        trig_map = {
            "sin": np.sin, "cos": np.cos,
            "tan": np.tan, "cot": lambda t: 1.0 / np.tan(t),
        }
        fn = trig_map.get(func_name, np.sin)
        y = A * fn(B * x + C) + D

        # Clip tan/cot discontinuities
        if func_name in ("tan", "cot"):
            y = np.where(np.abs(y) > (abs(A) * 10 + 2), np.nan, y)

        fig, ax = _graph_fig(figsize=(7, 4))
        ax.plot(x, y, "k-", linewidth=_LW, zorder=3)

        # Mark specified x-values
        for xm in mark_xs:
            try:
                ym = A * float(fn(B * float(xm) + C)) + D
                ax.plot(xm, ym, "ko", markersize=_DOT, zorder=5)
                ax.plot([xm, xm], [0, ym], "k--", linewidth=_LW_THIN, zorder=1)
            except Exception:
                pass

        # x-axis ticks at multiples of π/2
        x_ticks = [k * np.pi / 2 for k in range(int(x_max * 2 / np.pi) + 2)]
        x_ticks = [t for t in x_ticks if -0.05 <= t <= x_max + 0.2]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([_pi_label(t) for t in x_ticks], fontsize=8)

        # y-axis key ticks: amplitude multiples
        if abs(A) > 0:
            y_vals = sorted({0.0, A, -A, D, A + D, -A + D})
            ax.set_yticks([v for v in y_vals if abs(v) <= abs(A) * 2.5])

        ax.set_xlim(-0.3, x_max + 0.5)
        ylim = ax.get_ylim()
        _add_axis_arrows(ax)
        return fig, ax

    # ── Vector diagram ────────────────────────────────────────────────────────

    def _vector_diagram(self, p: dict):
        vectors = p.get("vectors", [])
        show_angle = p.get("show_angle", False)
        angle_label = p.get("angle_label", "α")
        show_sum = p.get("show_sum", False)

        fig, ax = _graph_fig()
        all_pts = [[0.0, 0.0]]

        for v in vectors:
            start = np.array(v.get("start", [0, 0]), dtype=float)
            end = np.array(v.get("end", [1, 1]), dtype=float)
            name = v.get("name", "v")
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>",
                                        color="black", lw=_LW,
                                        mutation_scale=14))
            mid = (start + end) / 2
            perp = _perp_offset(start, end, 0.2)
            ax.text(mid[0] + perp[0], mid[1] + perp[1],
                    f"$\\vec{{{name}}}$", fontsize=11, ha="center")
            all_pts.extend([start.tolist(), end.tolist()])

        # Angle arc between first two vectors
        if show_angle and len(vectors) >= 2:
            e1 = np.array(vectors[0].get("end", [1, 0]), dtype=float)
            e2 = np.array(vectors[1].get("end", [0, 1]), dtype=float)
            ang1 = math.degrees(math.atan2(e1[1], e1[0]))
            ang2 = math.degrees(math.atan2(e2[1], e2[0]))
            r_arc = min(np.linalg.norm(e1), np.linalg.norm(e2)) * 0.35
            arc = mpatches.Arc((0, 0), 2 * r_arc, 2 * r_arc,
                               angle=0,
                               theta1=min(ang1, ang2), theta2=max(ang1, ang2),
                               color="black", linewidth=_LW_THIN)
            ax.add_patch(arc)
            mid_ang = math.radians((ang1 + ang2) / 2)
            ax.text(r_arc * 1.25 * math.cos(mid_ang),
                    r_arc * 1.25 * math.sin(mid_ang),
                    angle_label, fontsize=_FS_MSR, ha="center")

        # Resultant
        if show_sum and vectors:
            ends = [np.array(v.get("end", [0, 0])) for v in vectors]
            total = sum(ends)
            ax.annotate("", xy=total, xytext=np.array([0, 0]),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=_LW_THIN, linestyle="dashed",
                                        mutation_scale=14))
            all_pts.append(total.tolist())

        # Limits
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = max(max(abs(x) for x in xs), max(abs(y) for y in ys), 1) * 0.35 + 0.7
        ax.set_xlim(-pad, max(xs) + pad)
        ax.set_ylim(-pad, max(ys) + pad)
        _add_axis_arrows(ax)
        return fig, ax

    # ── Sequence plot ─────────────────────────────────────────────────────────

    def _sequence_plot(self, p: dict):
        terms = p.get("terms", [1, 3, 5, 7, 9])

        fig, ax = _graph_fig(figsize=(6, 4))
        ns = list(range(1, len(terms) + 1))
        ax.scatter(ns, terms, color="black", s=35, zorder=4)
        ax.vlines(ns, 0, terms, colors="black", linewidth=_LW_THIN,
                  linestyle="dotted", zorder=1)

        for n, t in zip(ns, terms):
            ax.text(n, t + max(abs(max(terms) - min(terms)) * 0.04, 0.15),
                    _fmt_num(t), fontsize=8, ha="center", zorder=5)

        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=9)
        ax.set_xlim(0.3, len(ns) + 0.7)
        _add_axis_arrows(ax)
        ax.set_xlabel("")
        ax.text(len(ns) + 0.7, 0, "n", fontsize=10, ha="left", va="center")
        return fig, ax

    # ── Solid 3D ──────────────────────────────────────────────────────────────

    def _solid_3d(self, p: dict):
        solid_type = p.get("solid_type", "cube")
        dims = p.get("dimensions", {})
        dim_labels: dict = p.get("dimension_labels", {}) or {}
        labels: dict = p.get("labels", {}) or {}

        fig = plt.figure(figsize=(5, 5), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
        ax.set_axis_off()

        if solid_type == "cube":
            _draw_cube(ax, dims.get("side", 4), labels, dim_labels)
        elif solid_type == "rectangular_prism":
            _draw_prism(ax, dims.get("a", 4), dims.get("b", 3),
                        dims.get("h", 5), labels, dim_labels)
        elif solid_type == "pyramid":
            _draw_pyramid(ax, dims.get("base", 6), dims.get("height", 8),
                          labels, dim_labels)
        elif solid_type == "cylinder":
            _draw_cylinder(ax, dims.get("radius", 3), dims.get("height", 5), dim_labels)
        elif solid_type == "cone":
            _draw_cone(ax, dims.get("radius", 3), dims.get("height", 6), dim_labels)
        else:
            _draw_cube(ax, 4, {}, {})

        return fig, ax

    # ── Coordinate plane ──────────────────────────────────────────────────────

    def _coordinate_plane(self, p: dict):
        points_raw = p.get("points", [])
        segments = p.get("segments", [])
        polygon = p.get("polygon")
        seg_labels: dict = p.get("segment_labels", {}) or {}
        x_range = p.get("x_range", [-6, 6])
        y_range = p.get("y_range", [-6, 6])

        fig, ax = _graph_fig()
        pt_map = {d["label"]: np.array([d["x"], d["y"]], dtype=float)
                  for d in points_raw}

        # Polygon fill (very light hatch only)
        if polygon:
            poly_pts = [pt_map[lbl] for lbl in polygon if lbl in pt_map]
            if len(poly_pts) >= 3:
                poly_arr = np.array(poly_pts)
                ax.fill(poly_arr[:, 0], poly_arr[:, 1],
                        hatch="....", facecolor="none",
                        edgecolor="black", linewidth=0, alpha=0.3, zorder=1)
                closed = list(poly_pts) + [poly_pts[0]]
                closed_arr = np.array(closed)
                ax.plot(closed_arr[:, 0], closed_arr[:, 1],
                        "k-", linewidth=_LW, zorder=2)

        # Segments
        for seg in segments:
            if len(seg) == 2 and seg[0] in pt_map and seg[1] in pt_map:
                a, b = pt_map[seg[0]], pt_map[seg[1]]
                ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=_LW, zorder=2)
                key1 = seg[0] + seg[1]
                key2 = seg[1] + seg[0]
                lbl = seg_labels.get(key1) or seg_labels.get(key2)
                if lbl:
                    mid = (a + b) / 2
                    perp = _perp_offset(a, b, 0.25)
                    ax.text(mid[0] + perp[0], mid[1] + perp[1],
                            lbl, fontsize=_FS_MSR, ha="center", zorder=4)

        # Points
        for lbl, pt in pt_map.items():
            ax.plot(*pt, "ko", markersize=4, zorder=5)
            # Offset label away from origin
            off = 0.25
            ax.text(pt[0] + off, pt[1] + off, lbl,
                    fontsize=_FS_LBL, fontweight="bold", zorder=6)

        ax.set_xlim(x_range[0] - 0.4, x_range[1] + 0.6)
        ax.set_ylim(y_range[0] - 0.4, y_range[1] + 0.6)
        _clean_ticks(ax, x_range, y_range)
        _add_axis_arrows(ax)
        return fig, ax

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _placeholder(self, p: dict):
        fig, ax = _geometry_fig()
        ax.text(0.5, 0.5, "Figure", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        return fig, ax


# ── Geometry math helpers ─────────────────────────────────────────────────────

def _perp_offset(a: np.ndarray, b: np.ndarray, dist: float) -> np.ndarray:
    """Unit perpendicular vector to segment AB, scaled by dist."""
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([dist, 0.0])
    perp = np.array([-d[1], d[0]]) / n
    return perp * dist


def _foot_of_perpendicular(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Foot of perpendicular from point p to line AB."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    return a + t * ab


def _draw_right_angle_marker(ax, verts: np.ndarray, idx: int, size=0.25) -> None:
    v = verts[idx]
    prev_v = verts[(idx - 1) % 3]
    next_v = verts[(idx + 1) % 3]
    d1 = (prev_v - v) / np.linalg.norm(prev_v - v) * size
    d2 = (next_v - v) / np.linalg.norm(next_v - v) * size
    sq = np.array([v, v + d1, v + d1 + d2, v + d2])
    ax.plot(list(sq[:, 0]) + [sq[0, 0]],
            list(sq[:, 1]) + [sq[0, 1]],
            "k-", linewidth=_LW_THIN, zorder=3)


def _draw_right_angle_at(ax, corner: np.ndarray, dir1: np.ndarray,
                         dir2: np.ndarray, size=0.22) -> None:
    d1 = dir1 - corner
    if np.linalg.norm(d1) > 0:
        d1 = d1 / np.linalg.norm(d1) * size
    d2 = dir2 - corner
    if np.linalg.norm(d2) > 0:
        d2 = d2 / np.linalg.norm(d2) * size
    sq = np.array([corner, corner + d1, corner + d1 + d2, corner + d2])
    ax.plot(list(sq[:, 0]) + [sq[0, 0]],
            list(sq[:, 1]) + [sq[0, 1]],
            "k-", linewidth=_LW_THIN, zorder=3)


def _draw_angle_arc(ax, vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                    label: str, arc_frac=0.22) -> None:
    d1 = p1 - vertex
    d2 = p2 - vertex
    r = min(np.linalg.norm(d1), np.linalg.norm(d2)) * arc_frac
    ang1 = math.degrees(math.atan2(d1[1], d1[0]))
    ang2 = math.degrees(math.atan2(d2[1], d2[0]))
    # Always draw the smaller arc
    a_start = min(ang1, ang2)
    a_end = max(ang1, ang2)
    if a_end - a_start > 180:
        a_start, a_end = a_end, a_end + (360 - (a_end - a_start))
    arc = mpatches.Arc(vertex, 2 * r, 2 * r,
                       angle=0, theta1=a_start, theta2=a_end,
                       color="black", linewidth=_LW_THIN)
    ax.add_patch(arc)
    mid_ang = math.radians((a_start + a_end) / 2)
    ax.text(vertex[0] + r * 1.45 * math.cos(mid_ang),
            vertex[1] + r * 1.45 * math.sin(mid_ang),
            label, fontsize=_FS_MSR, ha="center", va="center", zorder=4)


def _incircle(v: np.ndarray):
    a = np.linalg.norm(v[1] - v[2])
    b = np.linalg.norm(v[0] - v[2])
    c = np.linalg.norm(v[0] - v[1])
    s = a + b + c
    if s < 1e-9:
        return v.mean(axis=0), 0.0
    incenter = (a * v[0] + b * v[1] + c * v[2]) / s
    area = 0.5 * abs(float(np.cross(v[1] - v[0], v[2] - v[0])))
    inradius = area / (s / 2)
    return incenter, inradius


def _circumcircle(v: np.ndarray):
    ax_, ay = v[0]
    bx, by = v[1]
    cx, cy = v[2]
    d = 2 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None, None
    ux = ((ax_**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax_**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax_ - cx) +
          (cx**2 + cy**2) * (bx - ax_)) / d
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(center - v[0]))


def _set_triangle_limits(ax, v: np.ndarray, pad=1.0) -> None:
    ax.set_xlim(v[:, 0].min() - pad, v[:, 0].max() + pad)
    ax.set_ylim(v[:, 1].min() - pad, v[:, 1].max() + pad)


# ── 3D solid drawing helpers ──────────────────────────────────────────────────

def _plot3_edge(ax, p1, p2, style="k-", lw=_LW):
    ax.plot3D(*zip(p1, p2), style, linewidth=lw)


def _label_midpoint_3d(ax, p1, p2, text, fontsize=8):
    mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
    ax.text(*mid, text, fontsize=fontsize, ha="center", zorder=5)


def _draw_cube(ax, s, labels, dim_labels):
    v = np.array([[0,0,0],[s,0,0],[s,s,0],[0,s,0],
                  [0,0,s],[s,0,s],[s,s,s],[0,s,s]], dtype=float)
    visible = [(0,1),(1,2),(3,2),(0,3),(4,5),(5,6),(7,6),(4,7),
               (0,4),(1,5),(2,6),(3,7)]
    hidden  = []
    for i, j in visible:
        _plot3_edge(ax, v[i], v[j])
    for i, j in hidden:
        _plot3_edge(ax, v[i], v[j], "k--", _LW_THIN)
    names = labels.get("vertices", list("ABCDA'B'C'D'"))
    for i, pt in enumerate(v):
        if i < len(names):
            ax.text(pt[0], pt[1], pt[2], names[i], fontsize=9, zorder=5)
    if "side" in dim_labels:
        _label_midpoint_3d(ax, v[0], v[1], dim_labels["side"])


def _draw_prism(ax, a, b, h, labels, dim_labels):
    v = np.array([[0,0,0],[a,0,0],[a,b,0],[0,b,0],
                  [0,0,h],[a,0,h],[a,b,h],[0,b,h]], dtype=float)
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]:
        _plot3_edge(ax, v[i], v[j])
    names = labels.get("vertices", list("ABCDA'B'C'D'"))
    for i, pt in enumerate(v):
        if i < len(names):
            ax.text(pt[0], pt[1], pt[2], names[i], fontsize=9, zorder=5)


def _draw_pyramid(ax, base, height, labels, dim_labels):
    h = base / 2
    bv = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], dtype=float)
    apex = np.array([0.0, 0.0, float(height)])
    for i in range(4):
        _plot3_edge(ax, bv[i], bv[(i+1) % 4])
        _plot3_edge(ax, bv[i], apex)
    base_names = labels.get("base_vertices", ["A","B","C","D"])
    for i, pt in enumerate(bv):
        if i < len(base_names):
            ax.text(pt[0], pt[1], pt[2]-0.3, base_names[i], fontsize=9, zorder=5)
    apex_lbl = labels.get("apex", "S")
    ax.text(*apex + np.array([0,0,0.3]), apex_lbl, fontsize=9, zorder=5)
    # Height line (dashed)
    foot = np.array([0.0, 0.0, 0.0])
    _plot3_edge(ax, apex, foot, "k--", _LW_THIN)
    if "height" in dim_labels:
        _label_midpoint_3d(ax, apex, foot, dim_labels["height"])
    if "base" in dim_labels:
        _label_midpoint_3d(ax, bv[0], bv[1], dim_labels["base"])


def _draw_cylinder(ax, r, h, dim_labels):
    theta = np.linspace(0, 2*np.pi, 80)
    ax.plot(r*np.cos(theta), r*np.sin(theta), np.zeros(80), "k-", linewidth=_LW)
    ax.plot(r*np.cos(theta), r*np.sin(theta), np.full(80, h), "k-", linewidth=_LW)
    for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax.plot3D([r*np.cos(t)]*2, [r*np.sin(t)]*2, [0, h], "k-", linewidth=_LW_THIN)
    if "radius" in dim_labels:
        ax.text(r/2, 0, h/2, dim_labels["radius"], fontsize=8)
    if "height" in dim_labels:
        ax.text(r+0.2, 0, h/2, dim_labels["height"], fontsize=8)


def _draw_cone(ax, r, h, dim_labels):
    theta = np.linspace(0, 2*np.pi, 80)
    ax.plot(r*np.cos(theta), r*np.sin(theta), np.zeros(80), "k-", linewidth=_LW)
    apex = [0, 0, h]
    for t in np.linspace(0, 2*np.pi, 8, endpoint=False):
        _plot3_edge(ax, [r*np.cos(t), r*np.sin(t), 0], apex)
    ax.text(0, 0, h+0.3, "S", fontsize=9, zorder=5)
    if "height" in dim_labels:
        ax.text(0.2, 0, h/2, dim_labels["height"], fontsize=8)


# ── Axis / tick helpers ───────────────────────────────────────────────────────

def _clean_ticks(ax, x_range, y_range) -> None:
    """Integer ticks only, no minor ticks, clean look."""
    x_ticks = [x for x in range(int(math.ceil(x_range[0])),
                                 int(math.floor(x_range[1])) + 1)
               if x != 0]
    y_ticks = [y for y in range(int(math.ceil(y_range[0])),
                                 int(math.floor(y_range[1])) + 1)
               if y != 0]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)


def _pi_label(t: float) -> str:
    frac = t / np.pi
    if abs(frac) < 0.01:
        return "0"
    if abs(frac - 1) < 0.01:
        return "π"
    if abs(frac - 2) < 0.01:
        return "2π"
    if abs(frac - 3) < 0.01:
        return "3π"
    if abs(frac - 4) < 0.01:
        return "4π"
    if abs(frac - 0.5) < 0.01:
        return "π/2"
    if abs(frac - 1.5) < 0.01:
        return "3π/2"
    return f"{frac:.1f}π"


def _fmt_num(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")
