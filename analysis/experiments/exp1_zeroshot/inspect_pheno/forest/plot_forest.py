#!/usr/bin/env python3
"""
CheXzero-Fig.3-style forest plot of per-phenotype zero-shot AUROC on the 86 applicable
INSPECT phenotypes: Ours (f2llm, black) with CT-CLIP (CT-RATE, light gray) overlaid on the
SAME rows (CT-CLIP drawn behind).

Layout (matches Tiu et al., Nat. Biomed. Eng. 2022, Fig. 3):
  * one row per phenotype, full name "<name> (n = <positives>)" right-aligned (wrapped to
    2 lines past WRAP_WIDTH chars); sharp 4-pointed-star marker + horizontal 95% CI whisker;
  * rows sorted by OURS SORT_BY (lower-CI-edge) descending; split across N_COLS columns;
    x-axis = AUC, 0.3 -> 1.0; each column = a wide label gutter + a thin data axes.

Data source (in priority order):
  1. forest/ci_<tag>.csv  (from bootstrap_ci.py)  -> bootstrap mean + 95% CI.
  2. fallback: point AUROC from results.json + analytic Hanley–McNeil CI.

Output: forest/forest_<tag>.png   (PNG only).   Usage:  python plot_forest.py [--tag f2llm]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the CheXzero Fig. 3 look: Helvetica/Arial sans-serif, pure black, italic math n.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",   # sans-serif math so the italic n matches the body
    "mathtext.default": "it",
    "text.color": "black", "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})

HERE = Path(__file__).resolve().parent
INSPECT = HERE.parent
EXP1 = INSPECT.parent
EXPERIMENTS = EXP1.parent                 # experiments/
ROOT = EXP1.parents[1]

LIST_86 = EXPERIMENTS / "exp4_confounder_audit" / "exp4_86_phenotypes.csv"
PERLABEL_CSV = INSPECT / "per_label_auc_all221_sorted.csv"

# CheXzero style: the highlighted model is black (drawn on top), the reference light-gray behind.
# Two plot modes (pick with --mode):
#   "ctclip"   : Ours f2llm zero-shot (black) vs CT-CLIP/CT-RATE (gray)  — the original figure.
#   "zs_vs_lp" : f2llm zero-shot (gray) vs f2llm linear probe on the full concept space, Adam
#                (black, drawn on top) — both arms f2llm, so the figure isolates the lift from
#                supervised linear probing over zero-shot. CIs: analytic Hanley–McNeil for BOTH
#                arms (consistent + reproducible offline; the CheXzero paired bootstrap needs the
#                linear probe's per-scan probs, which live cluster-side — see README).
F2LLM_ZS_RESULTS = ROOT / "outputs/v1/external/phenotype__test__f2llm__results.json"
# Adam linear probe = the 20 per-seed coefs at lr=3e-2 (NOT the *_20seeds_summary.csv, which is
# the L-BFGS arm). The plotted point is the per-label mean test AUROC over those 20 seeds.
ADAM_COEFS_DIR = ROOT / "outputs/v1/external"
ADAM_COEFS_GLOB = "phenotype__linear_f2llm__seed*__lr3e-2__coefs.pt"
N_VOL = 2612                     # INSPECT phenotype test volumes (y_221 rows) — for Hanley CIs

MODELS_BY_MODE = {
    "ctclip": {
        "models": {
            "ours_f2llm": dict(display="Ours (f2llm)", color="black", zorder=4,
                               results=F2LLM_ZS_RESULTS),
            "ctclip":     dict(display="CT-CLIP (CT-RATE)", color="#A9A9A9", zorder=3,
                               results=EXP1 / "ctclip/results/inspect_pheno__native__results.json"),
        },
        "order_by": "ours_f2llm",
    },
    "zs_vs_lp": {
        "models": {
            "lp": dict(display="Linear probe (f2llm)", color="black", zorder=4),
            "zs": dict(display="Zero-shot (f2llm)", color="#A9A9A9", zorder=3,
                       results=F2LLM_ZS_RESULTS),
        },
        "order_by": "lp",
    },
}
MODELS = MODELS_BY_MODE["ctclip"]["models"]   # rebound per-mode in plot(); module-level for helpers
ORDER_BY = "ours_f2llm"          # which model's score defines the row order (rebound per-mode)
SORT_BY = "lo"                   # rank rows by "auc" (point estimate) or "lo" (lower edge of 95% CI)
XLIM = (0.3, 1.0)
CHANCE_LINE = 0.5                # dashed reference line at AUC = 0.5 (chance); set None to disable
WRAP_WIDTH = 40                  # if "name (n = …)" exceeds this many chars, wrap to 2 lines
MARKER = (4, 1, 0)              # CheXzero Fig.3 marker = sharp 4-pointed star "✦"
MARKERSIZE = 14.0               # data-point size (larger)
ELINEWIDTH = 3.0                # 95% CI whisker thickness (larger)
MEW = 1.6                       # marker edge width (crisper larger stars)
# Layout: N columns, each = a WIDE phenotype-label gutter + a THIN AUC data axes (like the paper).
N_COLS = 3                       # number of columns (86 phenotypes split as evenly as possible)
FIG_WIDTH = 30.0                 # overall figure width (inches) — wider for more columns
ROW_HEIGHT = 0.66                # vertical inches per phenotype row (room for larger 2-line labels)
DATA_AXES_W = 0.14               # width of each thin AUC data axes (fraction of fig width)
COL_RIGHT_MARGIN = 0.025         # gap to the right of each column's data axes (fraction of fig width)
LABEL_FONTSIZE = 20              # phenotype labels (larger)
XTICK_FONTSIZE = 20              # x-axis numbers (larger)
XLABEL_FONTSIZE = 26             # "AUC" axis label (larger)
LEGEND_FONTSIZE = 28             # legend text (larger)


def _results_points(model: dict, names: list[str]) -> dict[str, float]:
    d = json.loads(Path(model["results"]).read_text())
    pla = d["per_label_auc"]
    pla = pla if isinstance(pla, dict) else dict(zip(d["labels"], pla))
    pla = {str(k).strip(): v for k, v in pla.items()}
    return {n: pla.get(n, np.nan) for n in names}


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int, z: float = 1.96):
    """Analytic 95% CI for AUROC from the point estimate + class counts only
    (Hanley & McNeil 1982). No per-sample labels needed — used as the fallback
    until y_221.npy is synced and the exact CheXzero bootstrap can run."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return np.nan, np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    se = np.sqrt(max(var, 0.0))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def _adam_linear_points(names: list[str]) -> tuple[dict[str, float], int]:
    """Per-label mean test AUROC of the f2llm full-concept-space linear probe (Adam, lr=3e-2),
    averaged over its 20 seeds. Read straight from the per-seed coefs' stored 'test_per_label_auc'
    (no torch forward pass needed — torch is only used to unpickle)."""
    import torch  # local import: only the linear-probe mode needs it
    files = sorted(ADAM_COEFS_DIR.glob(ADAM_COEFS_GLOB))
    if not files:
        raise SystemExit(f"no Adam linear-probe coefs match {ADAM_COEFS_DIR}/{ADAM_COEFS_GLOB}")
    acc: dict[str, list[float]] = {}
    for f in files:
        c = torch.load(f, map_location="cpu")
        pla = c["test_per_label_auc"]
        pla = pla if isinstance(pla, dict) else dict(zip(c["labels"], pla))
        for k, v in pla.items():
            acc.setdefault(str(k).strip(), []).append(float(v))
    return {n: float(np.mean(acc[n])) if n in acc else np.nan for n in names}, len(files)


def load_frame(tag: str, mode: str = "ctclip"):
    """Return (df, method). method in {'bootstrap','hanley','none'}.
    df cols: phenotype, n_pos, model, auc, lo, hi."""
    names = pd.read_csv(LIST_86)["phenotype"].str.strip().tolist()
    pl = pd.read_csv(PERLABEL_CSV); pl = pl.rename(columns={pl.columns[0]: "name"})
    npos_map = {str(k).strip(): int(v) for k, v in zip(pl["name"], pl["n_pos"])}

    if mode == "zs_vs_lp":
        # f2llm zero-shot vs f2llm linear probe (Adam); analytic Hanley–McNeil CIs for both arms.
        models = MODELS_BY_MODE["zs_vs_lp"]["models"]
        zs_pts = _results_points(models["zs"], names)
        lp_pts, n_seeds = _adam_linear_points(names)
        print(f"[zs_vs_lp] linear-probe arm = mean over {n_seeds} Adam (lr3e-2) seeds")
        pts_by_model = {"zs": zs_pts, "lp": lp_pts}
        rows = []
        for m, pts in pts_by_model.items():
            for n in names:
                npos = int(npos_map.get(n, 0) or 0)
                lo, hi = hanley_mcneil_ci(pts[n], npos, N_VOL - npos)
                rows.append(dict(phenotype=n, n_pos=npos, model=m, auc=pts[n], lo=lo, hi=hi))
        return pd.DataFrame(rows), "hanley"

    ci_csv = HERE / f"ci_{tag}.csv"
    if ci_csv.exists():                                   # exact CheXzero bootstrap
        df = pd.read_csv(ci_csv).rename(
            columns={"boot_mean": "auc", "ci_lower": "lo", "ci_upper": "hi"})
        df["phenotype"] = df["phenotype"].str.strip()
        return df[["phenotype", "n_pos", "model", "auc", "lo", "hi"]], "bootstrap"

    # fallback: point AUROC from results.json + analytic Hanley–McNeil CI
    rows = []
    for m, meta in MODELS.items():
        d = json.loads(Path(meta["results"]).read_text())
        n_vol = int(d.get("n_volumes", 2612))
        pts = _results_points(meta, names)
        for n in names:
            npos = int(npos_map.get(n, 0) or 0)
            lo, hi = hanley_mcneil_ci(pts[n], npos, n_vol - npos)
            rows.append(dict(phenotype=n, n_pos=npos, model=m,
                             auc=pts[n], lo=lo, hi=hi))
    return pd.DataFrame(rows), "hanley"


def _label(name: str, npos: int) -> str:
    """Full phenotype name + (n = …); wrap to 2 balanced lines if too long. Never truncate."""
    tag = f" ($n$ = {npos:,})"
    if len(name) + len(f" (n = {npos:,})") <= WRAP_WIDTH:
        return name + tag
    words = name.split()
    if len(words) < 2:
        return name + tag                      # single long token — leave as one line
    best = min(range(1, len(words)),
               key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return " ".join(words[:best]) + "\n" + " ".join(words[best:]) + tag


def plot(tag: str, mode: str = "ctclip", dpi: int = 200):
    models = MODELS_BY_MODE[mode]["models"]
    order_by = MODELS_BY_MODE[mode]["order_by"]
    df, method = load_frame(tag, mode)
    has_ci = method in ("bootstrap", "hanley")
    # row order = highlighted model, descending, by SORT_BY (lower-CI-edge by default, else point)
    sub = df[df.model == order_by].set_index("phenotype")
    skey = SORT_BY if (SORT_BY in sub.columns and sub[SORT_BY].notna().any()) else "auc"
    names = sub[skey].sort_values(ascending=False).index.tolist()
    npos = df.drop_duplicates("phenotype").set_index("phenotype")["n_pos"].to_dict()

    n_per = -(-len(names) // N_COLS)             # ceil → rows per column (86,3 -> 29/29/28)
    cols = [names[i * n_per:(i + 1) * n_per] for i in range(N_COLS)]
    nrow = max(len(c) for c in cols)

    fig = plt.figure(figsize=(FIG_WIDTH, ROW_HEIGHT * nrow + 1.8))
    # explicit thin data axes (wide label gutter to the left of each); no tight_layout so
    # the positions stick. bbox_inches='tight' at save grows the canvas to fit the labels.
    bottom, height = 0.05, 0.90
    slot = 1.0 / N_COLS                          # each column's horizontal slot
    col_lefts = [(i + 1) * slot - COL_RIGHT_MARGIN - DATA_AXES_W for i in range(N_COLS)]
    axes = [fig.add_axes([col_lefts[i], bottom, DATA_AXES_W, height]) for i in range(N_COLS)]

    for ax, col_names in zip(axes, cols):
        ax.set_xlim(*XLIM)
        ax.set_ylim(-0.6, len(col_names) - 0.4)
        ax.invert_yaxis()                         # best AUC on top
        if CHANCE_LINE is not None:               # dashed chance reference, behind the data
            ax.axvline(CHANCE_LINE, color="0.6", lw=1.2, dashes=(4, 3), zorder=0)
        yticks, ylabels = [], []
        for y, nm in enumerate(col_names):
            yticks.append(y)
            ylabels.append(_label(nm, npos.get(nm, 0)))   # full name, wrapped; italic n
            for m, meta in models.items():            # both series on the SAME line (same y)
                r = df[(df.phenotype == nm) & (df.model == m)]
                if r.empty or pd.isna(r["auc"].iloc[0]):
                    continue
                a = float(r["auc"].iloc[0]); zo = meta.get("zorder", 3)   # ours on top of gray
                if has_ci and not pd.isna(r["lo"].iloc[0]):
                    lo, hi = float(r["lo"].iloc[0]), float(r["hi"].iloc[0])
                    ax.errorbar(a, y, xerr=[[a - lo], [hi - a]], marker=MARKER,
                                markersize=MARKERSIZE, linestyle="none", color=meta["color"],
                                ecolor=meta["color"], mec=meta["color"], mew=MEW,
                                elinewidth=ELINEWIDTH, capsize=0, zorder=zo)   # capless, like the ref
                else:
                    ax.plot(a, y, marker=MARKER, markersize=MARKERSIZE, linestyle="none",
                            color=meta["color"], mec=meta["color"], mew=MEW, zorder=zo)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=LABEL_FONTSIZE, linespacing=1.0)
        # short y-tick at each row + outward x-ticks, like the reference
        ax.tick_params(axis="y", length=3.5, width=0.9, direction="out")
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE, length=4.0, width=0.9, direction="out")
        ax.set_xticks(np.arange(0.3, 1.01, 0.1))
        ax.set_xlabel("AUC", fontsize=XLABEL_FONTSIZE)    # ref says "AUC"
        for s in ("top", "right"):                        # keep left + bottom spines (L-axis)
            ax.spines[s].set_visible(False)

    # legend identifying the overlaid series (only when >1 model)
    if len(models) > 1:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker=MARKER, ls="none", ms=MARKERSIZE, mew=MEW,
                          color=meta["color"], mec=meta["color"], label=meta["display"])
                   for meta in models.values()]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.999),
                   handletextpad=0.5, columnspacing=2.5)

    suffix = "" if mode == "ctclip" else f"_{mode}"
    out = HERE / f"forest_{tag}{suffix}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # dump the exact plotted data alongside the figure for inspection / the paper table
    df[df.phenotype.isin(names)].to_csv(HERE / f"forest_{tag}{suffix}_data.csv", index=False)
    print(f"wrote {out}  (mode = {mode}; CI method = {method}; {len(names)} phenotypes; "
          f"models = {list(models)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f2llm")
    ap.add_argument("--mode", default="zs_vs_lp", choices=list(MODELS_BY_MODE),
                    help="zs_vs_lp: f2llm zero-shot vs f2llm linear probe (Adam); "
                         "ctclip: original Ours-f2llm vs CT-CLIP figure")
    ap.add_argument("--dpi", type=int, default=200)
    plot(**vars(ap.parse_args()))
