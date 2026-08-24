#!/usr/bin/env python3
"""2x4 grid of per-phenotype triptychs: AUROC gain | trusted concept audit | causal graph.

8 high-value phenotypes (pulmonary + cardiac/hemodynamic), ranked by trusted-
refinement AUROC gain. Each cell (2 columns x 4 rows) shows, left -> right:

  [1] AUROC difference  -- two bars, original-top-100 (before, tan) vs trusted
      (after, blue), each with 20 test-set-bootstrap AUROCs as scatter + s.d. bar.
      Reproduces refinement auc_before/auc_after exactly as the bar heights.
  [2] Trusted concept audit (blue) -- top-5 trusted concepts by the PLAIN
      importance llm_emb @ w_z (no 1/sd), mean-ranked, with 20 train-bootstrap
      values as scatter + s.d. bar.
  [3] Causal graph  -- red (proxy, blocked) and blue (direct) concept feeding the
      phenotype prediction. The red->prediction arrow is CUT (red X, no label);
      the blue->prediction arrow is the kept "direct signal". Big text inside circles.

Inputs (all local / on HF group `refinement`):
  refinement gains  experiments/exp4_confounder_audit/results/f2llm_refinement_all86_concepts.json
  trusted importance  outputs/v1/external/trusted_concept_probe_f2llm_lbfgs/
        trusted_f2llm_bootstrap_concept_importance_perseed_nosd.json   (build_trusted_perseed_nosd.py)
  AUROC bars        same dir: test__f2llm_{original_topk,trusted}__probs.npz (probs/labels/phecodes/volumes)
                    + perseed_coefs.npz (<slug>__y_test, test_volumes)

Output: experiments/exp4_confounder_audit/figures/top_gains_f2llm_grid2x4.png
"""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import roc_auc_score

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "text.color": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFINE = HERE / "results" / "f2llm_refinement_all86_concepts.json"
TDIR = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs"
NOSD = TDIR / "trusted_f2llm_bootstrap_concept_importance_perseed_nosd.json"
PERSEED = TDIR / "trusted_f2llm_bootstrap_perseed_coefs.npz"
INDEX = TDIR / "trusted_f2llm_bootstrap_perseed_index.json"
PROBS_AFTER = TDIR / "test__f2llm_trusted__probs.npz"          # trusted (after) test probs
SEED_GLOB = "outputs/v1/external/phenotype__linear_f2llm__seed{s}__lr3e-2__coefs.pt"  # full-bank Adam
OUT = HERE / "figures" / "top_gains_f2llm_grid2x4.png"
LEDGER = HERE / "results" / "fig6_matched_86_top20_ledger.json"
_L = json.loads(LEDGER.read_text())
_rows = [v for k, v in _L.items() if isinstance(v, list)][0]
LEDGER_CORR = {r["phenotype"]: r.get("procedural_exclusion_correction", 0.0) for r in _rows}

N_BOOT = 20
N_TOP = 5
NROWS, NCOLS = 4, 2

# colors
AUC_BEF, AUC_AFT = "#E0A06A", "#7FB3DD"      # tan before / light-blue after
BLUE, DBLUE = "#5B9BD5", "#2C6FAD"           # trusted bars
RED_FILL, RED_EDGE = "#F4DEDE", "#C0392B"
BLU_FILL, BLU_EDGE = "#DDEAF6", "#2C6FAD"
PRED_FILL = "#EDEDED"

# the 10 phenotypes (by phecode) + display name + curated graph themes (before/after).
# red = naive/proxy driver (gets blocked), blue = trusted/direct concept (kept).
PHENO = {
    "480.11": ("Pneumococcal pneumonia", "Diffuse ground-glass (COVID-like)", "Loculated effusion + atelectasis"),
    "797":    ("Shock", "Ascites / free fluid", "Active contrast extravasation"),
    "505":    ("Other pulmonary inflammation/edema", "Coalescing ground-glass", "Bilateral pleural effusion"),
    "420.2":  ("Pericarditis", "Pleural effusion", "Pericardial effusion"),
    "1013":   ("Asphyxia and hypoxemia", "Diffuse ground-glass + aortic atheroma", "Pneumothorax / atelectasis"),
    "428.1":  ("Congestive heart failure (NOS)", "Aortic / coronary calcification", "Bilateral pleural effusion"),
    "276.5":  ("Hypovolemia", "Biliary-duct dilatation", "Flat / collapsed IVC"),
    "501":    ("Aspiration pneumonitis", "Ground-glass + biliary-duct", "Subpleural nodular consolidation"),
}


def cap(s):
    return s[0].upper() + s[1:] if s else s


def wrap(c, w=24, n=3):
    return "\n".join(textwrap.wrap(cap(c), w)[:n])


# ----------------------------------------------------------------- data loaders
def load_auroc(refine_by_pc):
    """Full concept bank (Adam dense probe, 20-seed test AUC) vs trusted (20 test-bootstraps)."""
    za = np.load(PROBS_AFTER, allow_pickle=True)
    Pa = za["probs"]; pca = [str(x) for x in za["phecodes"]]
    nz = np.load(PERSEED, allow_pickle=True)
    idx = json.loads(INDEX.read_text())
    seed_auc = [torch.load(ROOT / SEED_GLOB.format(s=s), map_location="cpu",
                           weights_only=False)["test_per_label_auc"] for s in range(1, N_BOOT + 1)]
    out = {}
    for pc in PHENO:
        name = refine_by_pc[pc]["phenotype"]
        slug = idx[name]["slug"]
        y = nz[f"{slug}__y_test"].astype(int)
        fb = np.array([d[name] for d in seed_auc])              # 20 full-bank seed AUCs
        a = Pa[:, pca.index(pc)]
        N = len(y); ab = []
        for s in range(1, N_BOOT + 1):
            ridx = np.random.default_rng(1000 + s).integers(0, N, N)
            if 0 < y[ridx].sum() < len(ridx):
                ab.append(roc_auc_score(y[ridx], a[ridx]))       # trusted test-bootstrap
        # The stored trusted probabilities predate the procedural-exclusion fix, so
        # apply the same per-phenotype correction used for the ledger, Table S11 and
        # the manuscript text. Without this the bars would disagree with the text.
        corr = float(LEDGER_CORR.get(name, 0.0))
        out[pc] = {"before": float(fb.mean()), "after": float(roc_auc_score(y, a)) + corr,
                   "before_boot": fb, "after_boot": np.array(ab) + corr}
    return out


def load_trusted(refine_by_pc):
    """Top-5 trusted concepts (by bootstrap mean) per phenotype from the no-/sd file."""
    no = json.loads(NOSD.read_text())
    out = {}
    for pc in PHENO:
        name = refine_by_pc[pc]["phenotype"]
        pos = no[name]["trusted"]["positive"]
        top = sorted(pos, key=lambda c: -c["importance_boot_mean"])[:N_TOP]
        out[pc] = [{"concept": c["concept"], "mean": c["importance_boot_mean"],
                    "std": c["importance_boot_std"],
                    "per_seed": [float(x) for x in c.get("per_seed", [])]} for c in top]
    return out


# ----------------------------------------------------------------------- drawers
def draw_auroc(ax, d, rng):
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="0.9", lw=1.0, zorder=0)
    ax.bar([0, 1], [d["before"], d["after"]], color=[AUC_BEF, AUC_AFT], edgecolor="black",
           linewidth=1.0, width=0.74, zorder=2)
    for x, key in [(0, "before_boot"), (1, "after_boot")]:
        v = d[key]
        ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=10, facecolors="none",
                   edgecolors="0.2", linewidths=0.7, zorder=4)
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor="black",
                    elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full", "Refined"], fontsize=13)   # horizontal, centered under each bar
    ax.set_ylim(0.5, 1.0); ax.set_xlim(-0.6, 1.6)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("AUROC", fontsize=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_trusted(ax, rows, rng, xmax=10, xticks=(0, 2, 4, 6, 8, 10)):
    rows = sorted(rows, key=lambda r: -r["mean"])     # largest on top
    wrapped_labels = [wrap(r["concept"], 28, 3) for r in rows]
    line_counts = np.asarray([label.count("\n") + 1 for label in wrapped_labels])

    # Allocate each concept a vertical block based on its rendered line count.
    # Using physical (inch) coordinates keeps multiline labels separated, as in
    # the clinically aligned audit figure, instead of crowding equal y positions.
    label_size = 13
    label_line_spacing = 0.88
    line_height = label_size * label_line_spacing / 72.0
    block_heights = line_counts * line_height
    axes_height = ax.figure.get_figheight() * ax.get_position().height
    edge_padding = 0.08
    gap = (axes_height - block_heights.sum() - 2 * edge_padding) / (len(rows) - 1)
    if gap < 0.06:
        raise ValueError(f"concept labels do not fit without overlap; computed gap={gap:.3f} in")
    y = []
    cursor = edge_padding
    for block_height in block_heights:
        y.append(cursor + block_height / 2)
        cursor += block_height + gap
    y = np.asarray(y)
    mean = np.array([r["mean"] for r in rows]); std = np.array([r["std"] for r in rows])
    ax.set_axisbelow(True); ax.xaxis.grid(True, color="0.9", lw=1.0, zorder=0)
    ax.barh(y, mean, color=BLUE, edgecolor="black", linewidth=1.0, height=0.28, zorder=2)
    ax.errorbar(mean, y, xerr=std, fmt="none", ecolor="black", elinewidth=1.3,
                capsize=3.5, capthick=1.3, zorder=4)
    hivals = [float((mean + std).max())]
    for yi, r in zip(y, rows):
        v = r.get("per_seed", [])
        if v:
            vv = np.asarray(v); hivals.append(float(vv.max()))
            ax.scatter(vv, yi + rng.uniform(-0.07, 0.07, len(vv)), s=10, facecolors="none",
                       edgecolors="0.15", linewidths=0.7, zorder=5)
    ax.axvline(0, color="black", lw=1.4, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_labels, fontsize=label_size, linespacing=label_line_spacing)
    for t in ax.get_yticklabels():
        t.set_color("black")
    ax.tick_params(axis="y", length=3, width=1.0); ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(axes_height, 0)
    ax.set_xlim(0, xmax); ax.set_xticks(list(xticks))   # common scale across phenotypes (per-pheno override)
    ax.set_xlabel("Observation weight", fontsize=14)   # matches refinement_paired_audits.py
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_graph(ax, disp, before_th, after_th, asp):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    R = 0.20; ry = R * asp
    red, blu, pred = (0.245, 0.78), (0.245, 0.22), (0.77, 0.5)   # more vertical gap red<->blue

    def circle(c, txt, face, edge, tcol, w=13, fs=13):
        ax.add_patch(Ellipse(c, 2 * R, 2 * ry, facecolor=face, edgecolor=edge, lw=2.0,
                             zorder=3, clip_on=False))
        ax.text(c[0], c[1], wrap(txt, w, 4), ha="center", va="center", fontsize=fs,
                color=tcol, fontweight="bold", zorder=4)

    circle(red, before_th, RED_FILL, RED_EDGE, RED_EDGE)
    circle(blu, after_th, BLU_FILL, BLU_EDGE, BLU_EDGE)
    # A slash inside the display name ("inflammation/edema") gives textwrap no
    # break opportunity, so at w=13 it split words mid-token. Space the slash,
    # widen the wrap and drop a point so every line stays inside the ellipse.
    pred_txt = disp + " prediction"
    pred_w, pred_fs = 13, 13
    if "/" in pred_txt:
        pred_txt = pred_txt.replace("/", "/ ")
        pred_w, pred_fs = 16, 11
    circle(pred, pred_txt, PRED_FILL, "black", "black", w=pred_w, fs=pred_fs)

    # blue -> prediction : kept "direct signal"
    ax.annotate("", xy=(pred[0] - R + 0.01, pred[1] - 0.07), xytext=(blu[0] + R - 0.01, blu[1]),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=BLU_EDGE,
                                connectionstyle="arc3,rad=0.16"), zorder=2)

    # red -> prediction : BLOCKED (red X centred ON the arrow curve)
    x1, y1 = red[0] + R - 0.01, red[1]              # arrow tail
    x2, y2 = pred[0] - R + 0.01, pred[1] + 0.07     # arrow head
    rad = -0.16
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=RED_EDGE,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)
    # matplotlib Arc3 quadratic-Bezier control point -> t=0.5 midpoint ON the curve
    cx = (x1 + x2) / 2 + rad * (y2 - y1)
    cy = (y1 + y2) / 2 - rad * (x2 - x1)
    mx = 0.25 * x1 + 0.5 * cx + 0.25 * x2
    my = 0.25 * y1 + 0.5 * cy + 0.25 * y2
    r = 0.05                       # arm half-length (x-data units); aspect-comp -> visually square
    for base in (45, -45):
        a = np.radians(base + 30)  # rotate the X 30 deg so neither arm runs along the arrow
        dx, dy = r * np.cos(a), r * np.sin(a) * asp
        ax.plot([mx - dx, mx + dx], [my - dy, my + dy],
                color=RED_EDGE, lw=4.6, solid_capstyle="round", zorder=6)


def main():
    refine = json.loads(REFINE.read_text())
    by_pc = {str(r["phecode"]): r for r in refine}
    # Panel order is pinned to the published arrangement so that corrections to
    # the underlying values never reshuffle panel letters, which the manuscript
    # text references directly.
    order = ["480.11", "797", "505", "1013", "428.1", "276.5", "501", "420.2"]
    assert sorted(order) == sorted(PHENO), "panel set changed"
    print("order:", [(pc, PHENO[pc][0], round(by_pc[pc]["auc_gain"], 3)) for pc in order])

    auc = load_auroc(by_pc)
    trusted = load_trusted(by_pc)
    rng = np.random.default_rng(0)

    FIGW, FIGH = 23, 20.8   # 4 rows: scaled from 26 (5 rows) to keep panel size / graph aspect identical
    fig = plt.figure(figsize=(FIGW, FIGH))
    top_m, bot_m = 0.965, 0.020
    row_h = (top_m - bot_m) / NROWS
    inner_h = row_h * 0.60
    trusted_h = row_h * 0.70
    aw, tw, gw = 0.050, 0.105, 0.150
    ax_x, t_x, g_x = 0.038, 0.228, 0.340
    asp = (gw * FIGW) / (inner_h * FIGH)

    for k, pc in enumerate(order):
        col, row = divmod(k, NROWS)            # column-major data placement
        cx = col * 0.5
        y0 = top_m - (row + 1) * row_h + (row_h - inner_h) * 0.5
        disp, before_th, after_th = PHENO[pc]
        title_y = y0 + inner_h + row_h * 0.085
        # Nature-style panel letters follow visual reading order (left-to-right,
        # then top-to-bottom), even though the data are placed column-major.
        panel_i = row * NCOLS + col
        fig.text(cx + 0.005, title_y, chr(ord("a") + panel_i),
                 fontsize=30, fontweight="bold", family="Arial",
                 ha="left", va="bottom", color="black")
        fig.text(cx + 0.040, title_y,
                 disp, fontsize=21, fontweight="bold", va="bottom")
        draw_auroc(fig.add_axes([cx + ax_x, y0, aw, inner_h]), auc[pc], rng)
        trusted_y0 = y0 - (trusted_h - inner_h) / 2
        ax_t = fig.add_axes([cx + t_x, trusted_y0, tw, trusted_h])
        if pc == "480.11":   # Pneumococcal pneumonia: zoom per-concept weights to 0-3
            draw_trusted(ax_t, trusted[pc], rng, xmax=3, xticks=(0, 1, 2, 3))
        else:
            draw_trusted(ax_t, trusted[pc], rng)
        draw_graph(fig.add_axes([cx + g_x, y0, gw, inner_h]), disp, before_th, after_th, asp)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    OUT_PDF = OUT.with_suffix(".pdf")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
