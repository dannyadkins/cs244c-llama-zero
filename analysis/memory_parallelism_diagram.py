
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import build_small_config, estimate_num_parameters, human_readable_count

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

K = 12
N_D = 16
PARAM_BYTES = 2
GRAD_BYTES = 2
OPTIMIZER_STATE_BYTES = 12
BYTES_PER_GB = 1_000_000_000.0

MODEL_CONFIG = build_small_config()


def _compact_count(num_params: int) -> str:
    return f"{math.ceil(num_params / 1_000_000)}M"


def _format_gb(total_bytes: float) -> str:
    gb = total_bytes / BYTES_PER_GB
    rounded = round(gb, 1)
    if rounded.is_integer():
        return f"{int(rounded)}GB"
    return f"{rounded:.1f}GB"


def _shard_x(x: float, w: float, strip_w: float, placement: str) -> float:
    if placement == "left":
        return x
    if placement == "center":
        return x + ((w - strip_w) / 2.0)
    if placement == "right":
        return x + w - strip_w
    raise ValueError(f"Unsupported shard placement: {placement}")


psi = estimate_num_parameters(MODEL_CONFIG)
psi_label = _compact_count(psi)

baseline_bytes = (PARAM_BYTES + GRAD_BYTES + OPTIMIZER_STATE_BYTES) * psi
os_bytes = ((PARAM_BYTES + GRAD_BYTES) * psi) + ((OPTIMIZER_STATE_BYTES * psi) / N_D)
osg_bytes = (PARAM_BYTES * psi) + (((GRAD_BYTES + OPTIMIZER_STATE_BYTES) * psi) / N_D)
osgp_bytes = ((PARAM_BYTES + GRAD_BYTES + OPTIMIZER_STATE_BYTES) * psi) / N_D

blue = "#8FA2D2"
orange = "#E7AE7A"
green = "#9FC98B"
linec = "#6E88D0"
bg = "#F4F4F4"

fig, ax = plt.subplots(figsize=(14, 6.5), dpi=200)
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

x_label_div = 9.8
x_formula_div = 72
x_value_div = 89.2
row_lines = [81, 66, 48, 30, 12]

for y in row_lines[1:]:
    ax.plot([4, 98], [y, y], color=linec, lw=0.8)
for x in [x_label_div, x_formula_div, x_value_div]:
    ax.plot([x, x], [6, 96], color=linec, lw=0.8)

ax.text(19, 86, r"$\mathrm{GPU}_{0}$", fontsize=24, ha="center", va="center")
ax.text(41, 86, r"$\mathrm{GPU}_{i}$", fontsize=24, ha="center", va="center")
ax.text(63, 86, r"$\mathrm{GPU}_{N-1}$", fontsize=24, ha="center", va="center")
ax.text(81, 90, "Memory\nConsumed", fontsize=24, ha="center", va="center")
ax.text(93.6, 91, f"K={K}\nΨ={psi_label}\nN$_d$={N_D}", fontsize=16, ha="center", va="center")

ax.text(4.4, 74.5, "Baseline", fontsize=24, ha="center", va="center")
ax.text(4.0, 56, r"$\mathrm{P}_{\mathrm{os}}$", fontsize=24, ha="center", va="center")
ax.text(4.3, 38, r"$\mathrm{P}_{\mathrm{os{+}g}}$", fontsize=24, ha="center", va="center")
ax.text(4.6, 20, r"$\mathrm{P}_{\mathrm{os{+}g{+}p}}$", fontsize=24, ha="center", va="center")

ax.text(81, 73, r"$(2 + 2 + K)\,*\,\Psi$", fontsize=24, ha="center", va="center")
ax.text(94, 73, _format_gb(baseline_bytes), fontsize=18, ha="center", va="center")
ax.text(81, 56, r"$2\Psi\,+\,2\Psi\,+\,\frac{K\,*\,\Psi}{N_d}$", fontsize=24, ha="center", va="center")
ax.text(94, 56, _format_gb(os_bytes), fontsize=18, ha="center", va="center")
ax.text(81, 38, r"$2\Psi\,+\,\frac{(2{+}K)*\Psi}{N_d}$", fontsize=24, ha="center", va="center")
ax.text(94, 38, _format_gb(osg_bytes), fontsize=18, ha="center", va="center")
ax.text(81, 20, r"$\frac{(2 + 2 + K)*\Psi}{N_d}$", fontsize=24, ha="center", va="center")
ax.text(94, 20, _format_gb(osgp_bytes), fontsize=18, ha="center", va="center")

def draw_baseline(x, y, w=16, h_opt=9, h_grad=1.3, h_param=1.3):
    ax.add_patch(Rectangle((x, y), w, h_opt, facecolor=green, edgecolor="none"))
    ax.add_patch(Rectangle((x, y+h_opt), w, h_grad, facecolor=orange, edgecolor="none"))
    ax.add_patch(Rectangle((x, y+h_opt+h_grad), w, h_param, facecolor=blue, edgecolor="none"))

def draw_os(x, y, w=16, h_opt=9, h_grad=1.3, h_param=1.3, strip_w=1.1, placement="left"):
    strip_x = _shard_x(x, w, strip_w, placement)
    ax.add_patch(Rectangle((x, y+h_opt+h_grad), w, h_param, facecolor=blue, edgecolor="none"))
    ax.add_patch(Rectangle((x, y+h_opt), w, h_grad, facecolor=orange, edgecolor="none"))
    ax.add_patch(Rectangle((strip_x, y), strip_w, h_opt, facecolor=green, edgecolor="none"))

def draw_osg(x, y, w=16, h_opt=9, h_grad=1.3, h_param=1.3, strip_w=1.1, placement="left"):
    strip_x = _shard_x(x, w, strip_w, placement)
    ax.add_patch(Rectangle((x, y+h_opt+h_grad), w, h_param, facecolor=blue, edgecolor="none"))
    ax.add_patch(Rectangle((strip_x, y+h_opt), strip_w, h_grad, facecolor=orange, edgecolor="none"))
    ax.add_patch(Rectangle((strip_x, y), strip_w, h_opt, facecolor=green, edgecolor="none"))

def draw_osgp(x, y, w=16, h_opt=9, h_grad=1.3, h_param=1.3, strip_w=1.1, placement="left"):
    strip_x = _shard_x(x, w, strip_w, placement)
    ax.add_patch(Rectangle((strip_x, y), strip_w, h_opt, facecolor=green, edgecolor="none"))
    ax.add_patch(Rectangle((strip_x, y+h_opt), strip_w, h_grad, facecolor=orange, edgecolor="none"))
    ax.add_patch(Rectangle((strip_x, y+h_opt+h_grad), strip_w, h_param, facecolor=blue, edgecolor="none"))

gpu_xs = [10.5, 32.5, 54.5]
ellipsis_xs = [28.5, 50.5]
y_positions = {"baseline": 69.5, "os": 51.0, "osg": 33.0, "osgp": 15.5}
gpu_shard_placements = ["left", "center", "right"]

for x, placement in zip(gpu_xs, gpu_shard_placements):
    draw_baseline(x, y_positions["baseline"])
    draw_os(x, y_positions["os"], placement=placement)
    draw_osg(x, y_positions["osg"], placement=placement)
    draw_osgp(x, y_positions["osgp"], placement=placement)

for ex in ellipsis_xs:
    for y in [76.0, 57.5, 39.5, 21.5]:
        ax.text(ex, y, "...", fontsize=30, ha="center", va="center")

legend_y = 2.5
legend_items = [("Parameters", blue, 14), ("Gradients", orange, 38), ("Optimizer States", green, 60)]
for label, color, x in legend_items:
    ax.add_patch(Rectangle((x, legend_y - 0.8), 2.2, 3.2, facecolor=color, edgecolor="none"))
    ax.text(x + 3.0, legend_y + 0.7, label, fontsize=24, ha="left", va="center")

plt.tight_layout(pad=0.15)
fig.savefig("memory_parallelism_diagram.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
