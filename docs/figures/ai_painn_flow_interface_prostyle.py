"""Final professional figure for PaiNN + Flow Matching + Interface generation.

Design priorities:
1) Math indices complete (sub/superscripts)
2) No text/line overlap
3) Reference-like professional architecture style
4) Arial font for publication compatibility
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "savefig.dpi": 600,
        "text.usetex": False,
        "mathtext.default": "it",
    }
)


BG = "#ECECEC"
PANEL_BG = "#F3F4F6"
EDGE = "#111111"


def rbox(ax, x, y, w, h, fc="#FFFFFF", ec=EDGE, lw=1.5, rs=0.014):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={rs}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def text_center(ax, x, y, s, fs=12, w="normal", c="#111111"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, fontweight=w, color=c, transform=ax.transAxes, zorder=3)


def text_left(ax, x, y, s, fs=10, c="#111111"):
    ax.text(x, y, s, ha="left", va="top", fontsize=fs, color=c, transform=ax.transAxes, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.4, ms=10):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(a)


def dlink(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], transform=ax.transAxes, linestyle=(0, (4, 4)), linewidth=1.4, color="#9CA3AF", zorder=1)


def mini_crystal(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x, y), 0.05 * s, 0.05 * s, facecolor="#EFF6FF", edgecolor="#93C5FD", linewidth=0.8, transform=ax.transAxes, zorder=2))
    pts = [
        (x + 0.006 * s, y + 0.006 * s),
        (x + 0.044 * s, y + 0.006 * s),
        (x + 0.006 * s, y + 0.044 * s),
        (x + 0.044 * s, y + 0.044 * s),
        (x + 0.025 * s, y + 0.025 * s),
    ]
    for i, (px, py) in enumerate(pts):
        col = "#DC2626" if i < 4 else "#2563EB"
        ax.add_patch(Circle((px, py), 0.0034 * s, facecolor=col, edgecolor="#FCA5A5", linewidth=0.6, transform=ax.transAxes, zorder=3))


def add_formula_block(ax, x, y, w, h, title, lines, fc="#FFFFFF"):
    rbox(ax, x, y, w, h, fc=fc, ec="#374151", lw=1.2, rs=0.012)
    text_left(ax, x + 0.012, y + h - 0.015, title, fs=10.5, c="#111827")
    cy = y + h - 0.055
    for line in lines:
        text_left(ax, x + 0.014, cy, line, fs=9.5, c="#1F2937")
        cy -= 0.035


def build_figure(figsize=(18, 8.6)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Title
    text_left(
        ax,
        0.02,
        0.98,
        "PaiNN + Flow Matching Framework for Fullerene and Interface Structure Generation",
        fs=18,
        c="#0F172A",
    )

    # ---------------- Left: Short-range PaiNN ----------------
    rbox(ax, 0.025, 0.105, 0.31, 0.80, fc=PANEL_BG, ec=EDGE, lw=2.0, rs=0.025)
    text_left(ax, 0.035, 0.875, "Short-range (Local Equivariant Modeling)", fs=16.5)
    text_center(ax, 0.18, 0.12, "PaiNN Message + Update Blocks", fs=13.5)

    mini_crystal(ax, 0.04, 0.61, 1.35)

    # input projections
    rbox(ax, 0.07, 0.29, 0.065, 0.055, fc="#E5E7EB")
    rbox(ax, 0.15, 0.29, 0.065, 0.055, fc="#E5E7EB")
    rbox(ax, 0.23, 0.29, 0.065, 0.055, fc="#E5E7EB")
    text_center(ax, 0.102, 0.317, "Linear", fs=11)
    text_center(ax, 0.182, 0.317, "Linear", fs=11)
    text_center(ax, 0.262, 0.317, "Linear", fs=11)

    text_center(ax, 0.102, 0.255, r"$s_i^{(l)}$", fs=15)
    text_center(ax, 0.182, 0.255, r"$V_j^{(l)}$", fs=15)
    text_center(ax, 0.262, 0.255, r"$\hat{r}_{ij}, d_{ij}$", fs=15)

    arrow(ax, 0.102, 0.273, 0.102, 0.29)
    arrow(ax, 0.182, 0.273, 0.182, 0.29)
    arrow(ax, 0.262, 0.273, 0.262, 0.29)

    # attention/message stack
    rbox(ax, 0.095, 0.40, 0.215, 0.15, fc="#C4B5FD", rs=0.018)
    text_center(ax, 0.203, 0.475, "Equivariant Message", fs=15)
    text_center(ax, 0.203, 0.44, r"$m_{ij}=W_s(d_{ij})s_j + W_v(d_{ij})(V_j\cdot\hat{r}_{ij})$", fs=10.5)

    rbox(ax, 0.145, 0.57, 0.115, 0.05, fc="#FDE68A")
    text_center(ax, 0.202, 0.595, r"Aggregate $\sum_j m_{ij}$", fs=11.2)

    rbox(ax, 0.15, 0.645, 0.105, 0.05, fc="#E5E7EB")
    text_center(ax, 0.202, 0.67, r"AdaLN$(t,C)$", fs=11.2)

    rbox(ax, 0.14, 0.72, 0.125, 0.055, fc="#FDE68A")
    text_center(ax, 0.202, 0.747, r"Update $s_i,V_i$", fs=12)

    arrow(ax, 0.202, 0.345, 0.202, 0.40)
    arrow(ax, 0.202, 0.55, 0.202, 0.57)
    arrow(ax, 0.202, 0.62, 0.202, 0.645)
    arrow(ax, 0.202, 0.695, 0.202, 0.72)
    arrow(ax, 0.202, 0.775, 0.202, 0.915)
    text_center(ax, 0.204, 0.902, r"$s_i^{(l+1)},\ V_i^{(l+1)}$", fs=14)

    # ---------------- Center: Flow Matching ----------------
    rbox(ax, 0.36, 0.22, 0.22, 0.56, fc=PANEL_BG, ec=EDGE, lw=2.0, rs=0.025)

    rbox(ax, 0.415, 0.79, 0.11, 0.065, fc="#FBD38D")
    text_center(ax, 0.47, 0.822, r"MLP: $v_\theta$", fs=14)

    rbox(ax, 0.39, 0.70, 0.16, 0.055, fc="#E7EAB9")
    text_center(ax, 0.47, 0.728, "Fusion of t and C", fs=14)

    rbox(ax, 0.39, 0.595, 0.16, 0.08, fc="#7DD3FC")
    text_center(ax, 0.47, 0.64, "Flow Matching Core", fs=15)

    rbox(ax, 0.405, 0.48, 0.13, 0.08, fc="#F5D29A")
    text_center(ax, 0.47, 0.525, "PaiNN x L layers", fs=14)

    rbox(ax, 0.415, 0.365, 0.11, 0.08, fc="#F3DDE3")
    text_center(ax, 0.47, 0.405, "Graph Embedding", fs=14)

    # arrows center stack
    arrow(ax, 0.47, 0.755, 0.47, 0.79)
    arrow(ax, 0.47, 0.675, 0.47, 0.70)
    arrow(ax, 0.47, 0.56, 0.47, 0.595)
    arrow(ax, 0.47, 0.445, 0.47, 0.48)
    arrow(ax, 0.47, 0.855, 0.47, 0.955)

    text_center(ax, 0.47, 0.94, r"$\hat{v}_{\theta}(x_t,t,C)$", fs=14)
    text_center(ax, 0.47, 0.338, r"Inputs: $x_t,\ t,\ C,\ E$", fs=13)

    # Center math block (key to algorithm core)
    add_formula_block(
        ax,
        0.365,
        0.145,
        0.21,
        0.095,
        "Flow Matching equations",
        [
            r"$x_t=(1-t)x_0+t\epsilon,\ \ t\sim\mathcal{U}(0,1)$",
            r"$v_{target}=\epsilon-x_0$",
            r"$\mathcal{L}_{FM}=\mathbb{E}\|v_\theta(x_t,t,C)-v_{target}\|_2^2$",
        ],
        fc="#FFFFFF",
    )

    # ---------------- Right: Long-range + Interface ----------------
    rbox(ax, 0.605, 0.105, 0.37, 0.80, fc=PANEL_BG, ec=EDGE, lw=2.0, rs=0.025)
    text_left(ax, 0.615, 0.875, "Long-range + Interface Construction", fs=16)
    text_center(ax, 0.79, 0.122, "Pseudo-Particle Diffraction + Physical Assembly", fs=13)

    rbox(ax, 0.635, 0.655, 0.315, 0.07, fc="#E5E7EB")
    text_center(ax, 0.792, 0.69, "Structure Factors $F(\mathbf{H})$", fs=19)

    rbox(ax, 0.645, 0.545, 0.295, 0.07, fc="#EFD5DA")
    text_center(ax, 0.792, 0.58, "Broadcast and Coupling", fs=19)

    rbox(ax, 0.67, 0.405, 0.075, 0.06, fc="#E5E7EB")
    rbox(ax, 0.81, 0.405, 0.075, 0.06, fc="#E5E7EB")
    text_center(ax, 0.707, 0.435, "MLP", fs=15)
    text_center(ax, 0.847, 0.435, "MLP", fs=15)

    rbox(ax, 0.675, 0.755, 0.235, 0.06, fc="#FDE68A")
    text_center(ax, 0.792, 0.785, r"$\mathrm{Re}(F)\oplus\mathrm{Im}(F)$", fs=17)

    text_center(ax, 0.665, 0.51, r"$f_i^*(\mathbf{H})$", fs=15)
    text_center(ax, 0.80, 0.51, r"$f_j^*(\mathbf{H})$", fs=15)
    text_center(ax, 0.69, 0.325, r"$h_i^{(L)}$", fs=16)
    text_center(ax, 0.83, 0.325, r"$h_j^{(L)}$", fs=16)
    text_center(ax, 0.81, 0.83, r"$F_{concat}$", fs=17)

    # hkl tags
    for x, tag in [(0.615, "010"), (0.665, "101"), (0.715, "002"), (0.83, "014")]:
        rbox(ax, x, 0.60, 0.028, 0.04, fc="#E9D5FF", rs=0.018, lw=1.2)
        text_center(ax, x + 0.014, 0.62, tag, fs=9)

    mini_crystal(ax, 0.89, 0.615, 1.1)
    mini_crystal(ax, 0.655, 0.255, 0.92)

    arrow(ax, 0.707, 0.465, 0.707, 0.545)
    arrow(ax, 0.847, 0.465, 0.847, 0.545)
    arrow(ax, 0.685, 0.545, 0.685, 0.655)
    arrow(ax, 0.742, 0.545, 0.742, 0.655)
    arrow(ax, 0.800, 0.545, 0.800, 0.655)
    arrow(ax, 0.857, 0.545, 0.857, 0.655)
    arrow(ax, 0.792, 0.725, 0.792, 0.755)
    arrow(ax, 0.792, 0.815, 0.792, 0.955)

    text_center(ax, 0.792, 0.94, r"$F_{concat}\rightarrow \hat{y}$", fs=14)

    # Right math block for sampling
    add_formula_block(
        ax,
        0.615,
        0.145,
        0.35,
        0.115,
        "Long-range principle + sampling/interface pipeline",
        [
            r"$F(\mathbf{H})=\sum_j f_j^*(\mathbf{H})\exp\left(i2\pi\mathbf{H}\cdot\mathbf{r}_j\right)$",
            r"$g_{ij}=\phi\!\left([h_i^{(L)},h_j^{(L)},\mathrm{Re}(F),\mathrm{Im}(F)]\right)$",
            r"$x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t,C)$ (Euler, 50 steps)",
            r"Project + relax + validity check $\rightarrow$ slab adsorption $\rightarrow$ interface",
        ],
        fc="#FFFFFF",
    )

    # cross-panel links
    dlink(ax, 0.335, 0.84, 0.36, 0.27)
    dlink(ax, 0.58, 0.59, 0.605, 0.84)
    arrow(ax, 0.335, 0.56, 0.392, 0.62)
    arrow(ax, 0.605, 0.62, 0.55, 0.62)

    return fig


def main():
    out_dir = Path(__file__).resolve().parent
    fig = build_figure()
    stem = out_dir / "ai_painn_flow_interface_prostyle_refined_v2"

    fig.savefig(f"{stem}.png", dpi=600, facecolor=BG, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", facecolor=BG, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"- {stem}.png")
    print(f"- {stem}.pdf")
    print(f"- {stem}.svg")


if __name__ == "__main__":
    main()
