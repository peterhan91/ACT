#!/usr/bin/env python3
"""Top-8 AUROC-gain phenotypes: concept auditing WITH vs WITHOUT concept intervention.

For the 8 INSPECT phenotypes with the largest concept-purification AUROC gain
(f2llm, auc_after - auc_before from f2llm_refinement_all86_concepts.json), each
row shows three things left -> right:

  [1] WITHOUT intervention  -- naive concept audit of the full-bank Adam linear
      probe: raw top-5 drivers by 20-seed mean coefficient
      coef = W[label] . (centered+L2-normalized concept embedding), 20 per-seed
      values as scatter. (Same construction as plot_shortcut.py.)        [red]
  [2] WITH intervention  -- concept audit of the trusted-subset LBFGS probe
      (purified concept space). Bars show mean +- s.d. of probe importance over 20
      train-set-bootstrap resamples (default_rng(seed).integers(0,N,N), seeds 1..20)
      from trusted_f2llm_bootstrap_concept_importance.json; the top-5 are picked and
      ordered by the bootstrap MEAN importance (matching the naive panel's mean
      ranking; the s.d. error bar still shows each concept's spread). Per-seed scatter is drawn
      when the *_perseed.json (recompute_trusted_bootstrap_perseed.py) is present. [blue]
  [3] Graph model  -- naive driver (red) and trusted refined concept (blue) both
      feeding the phenotype prediction, annotated AUROC before -> after (+gain).

Probe bundles : outputs/v1/external/phenotype__linear_f2llm__seed{1..20}__lr3e-2__coefs.pt (Adam)
Concept bank  : concept_bank.f2llm_emb.npz (376k concepts, D=5120)
Trusted audit : outputs/v1/external/trusted_concept_probe_f2llm_lbfgs/trusted_f2llm_bootstrap_concept_importance.json
                (20 train-bootstrap seeds; importance_point + importance_boot_mean/std per concept)
Refinement    : experiments/exp4_confounder_audit/results/f2llm_refinement_all86_concepts.json

Phenotype selection + AUROCs/gains are read live from the refinement JSON; bar
concepts/scores are read live from the probe artifacts. Short circle phrases in the
graph are curated from naive_driver_theme + after_top6_pos_concepts.

The naive-audit recompute (full 7.7 GB bank pass) is cached to results/_top_gains_naive_cache.json;
delete that file or pass --recompute to rebuild.

Output: experiments/exp4_confounder_audit/figures/top_gains_f2llm.png
"""
from __future__ import annotations
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "text.color": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFINE = HERE / "results" / "f2llm_refinement_all86_concepts.json"
# Bootstrap file carries importance_point (== the old deterministic `score`) plus
# importance_boot_mean/std over 20 train-resample seeds, so it fully replaces the
# single-fit trusted_f2llm_concept_importance.json the middle plot used to read.
TRUSTED_BOOT = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs/trusted_f2llm_bootstrap_concept_importance.json"
# Optional richer file written by recompute_trusted_bootstrap_perseed.py: same schema
# but each concept also carries a `per_seed` list of the 20 raw bootstrap importances.
# When present it is preferred so the middle bars get per-seed scatter (like the left).
TRUSTED_BOOT_PERSEED = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs/trusted_f2llm_bootstrap_concept_importance_perseed.json"
BANK_NPZ = ROOT / "concept_bank.f2llm_emb.npz"
SEED_GLOB = "outputs/v1/external/phenotype__linear_f2llm__seed{s}__lr3e-2__coefs.pt"
CACHE = HERE / "results" / "_top_gains_naive_cache.json"
OUT = HERE / "figures" / "top_gains_f2llm.png"

# --nosd : trusted importance = llm_emb @ w_z (plain dot, WITHOUT the 1/sd z-score
# de-standardization). Concepts are re-selected + re-scored under that metric; the
# values are O(1) (not ~1e3), so the axis unit drops the x10^3. Naive/graph unchanged.
# --ctpa : restrict the panels to the curated high-value CTPA phenotypes (CTPA_PHECODES),
# ranked by trusted-refinement AUROC gain, positive-gain only (both performance AND the
# top-5 trusted concepts are clinically clean). Otherwise: global top-8 by gain.
NOSD = "--nosd" in sys.argv
CTPA = "--ctpa" in sys.argv
TRUST_SCALE = 1.0 if NOSD else 1000.0
TRUST_UNIT = "" if NOSD else r" ($\times10^{3}$)"
if NOSD:
    TRUSTED_BOOT_PERSEED = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs/trusted_f2llm_bootstrap_concept_importance_perseed_nosd.json"
_suffix = ("_ctpa" if CTPA else "") + ("_nosd" if NOSD else "")
if _suffix:
    OUT = HERE / "figures" / f"top_gains_f2llm{_suffix}.png"

# High-value CTPA phenotypes (radiologist-curated, clinically clean top-5 audit).
# Phecodes from the user's table-1; --ctpa keeps the in-86 ones with positive gain.
CTPA_PHECODES = {"507", "503", "505", "509.1", "480", "480.1", "509.2", "501",
                 "508", "496.3", "1013", "480.11", "516", "516.1", "519", "514"}

N_SEEDS = 20
N_TOP = 5
N_PANELS = 8
TCRIT_19 = 2.093  # t_{0.975, df=19}

RED, BLUE, CIRC = "#CB8A8A", "#5B9BD5", "#EDEDED"   # salmon naive / blue trusted
DRED, DBLUE = "#C0392B", "#2C6FAD"

# Concise before/after theme phrases for the graph circles, curated from each row's
# naive_driver_theme + after_top6_pos_concepts (AUROC numbers are NOT hardcoded).
THEMES = {
    "Pneumococcal pneumonia":
        ("Diffuse ground-glass /\nCOVID-like opacity", "Loculated pleural effusion\n+ atelectasis"),
    "Respiratory insufficiency":
        ("Widespread ground-glass\nconsolidation", "Focal nodular\nconsolidation"),
    "Other disorders of liver":
        ("Pericardial / pleural\neffusion", "Hepatic steatosis /\nbiliary tree"),
    "Shock":
        ("Ascites / free\nabdominal fluid", "Active contrast\nextravasation"),
    "Anemia in neoplastic disease":
        ("Pneumonic\nconsolidation", "Lymphadenopathy /\nnodal mass"),
    "Diaphragmatic hernia":
        ("Aortic / coronary\ncalcification", "Hiatal hernia\n(stomach herniation)"),
    "Other pulmonary inflamation or edema":
        ("Coalescing ground-glass\nconsolidation", "Bilateral pleural\neffusion / edema"),
    "End stage renal disease":
        ("Aortic / coronary\ncalcification", "Renal calcification /\nstone"),
}
# CTPA-figure themes, curated from each row's before (prior_top6 / naive_driver) vs
# after (after_top6) concepts — several show a real confounder->clean transition
# (505 sheds aortic/cardiac, 1013 sheds aortic atheroma, 501 sheds biliary-duct).
CTPA_THEMES = {
    "Pneumococcal pneumonia":
        ("Diffuse ground-glass /\nCOVID-like opacity", "Loculated pleural effusion\n+ compressive atelectasis"),
    "Respiratory insufficiency":
        ("Widespread COVID-like\nground-glass", "Focal nodular\nconsolidation"),
    "Other pulmonary inflamation or edema":
        ("Coalescing ground-glass\n(+ aortic / cardiac proxy)", "Bilateral pleural\neffusion"),
    "Asphyxia and hypoxemia":
        ("Diffuse ground-glass\n(+ aortic atheroma proxy)", "Pneumothorax / atelectasis /\ninfiltration"),
    "Bronchiectasis":
        ("Bronchiectasis +\nnodular consolidation", "Localized bronchial\nwall thickening"),
    "Pneumonitis due to inhalation of food or vomitus":
        ("Ground-glass\n(+ biliary-duct proxy)", "Subpleural nodular\nconsolidation"),
    "Pneumonia":
        ("Widespread COVID-like\nground-glass", "Nodular / lingular\nconsolidation"),
    "Pleurisy; pleural effusion":
        ("Effusion + adjacent\natelectasis", "Pleural effusion /\nhydropneumothorax"),
}
DISPLAY = {
    "Other pulmonary inflamation or edema": "Other pulmonary inflammation/edema",
    "Pneumonitis due to inhalation of food or vomitus": "Aspiration pneumonitis",
    "Pleurisy; pleural effusion": "Pleurisy / pleural effusion",
}


def get_themes(ph, r):
    """Curated CTPA themes -> legacy THEMES -> data-derived (naive_driver + top after-concept)."""
    if ph in CTPA_THEMES:
        return CTPA_THEMES[ph]
    if ph in THEMES:
        return THEMES[ph]
    before = wrap(r.get("naive_driver_theme") or "naive driver", 22, 2)
    after_list = r.get("after_top6_pos_concepts") or []
    after = wrap(after_list[0], 22, 2) if after_list else "trusted concept"
    return (before, after)


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def wrap(c: str, w: int = 30, n: int = 3) -> str:
    return "\n".join(textwrap.wrap(cap(c), w)[:n])


# --------------------------------------------------------------------- naive audit
def compute_naive(label_order):
    """Recompute full-bank Adam 20-seed top-5 drivers + per-seed scatter + AUROC CI."""
    import torch
    Ws, aucs, labels_ref, idx = [], {p: [] for p in label_order}, None, None
    for s in range(1, N_SEEDS + 1):
        b = torch.load(ROOT / SEED_GLOB.format(s=s), map_location="cpu", weights_only=False)
        if labels_ref is None:
            labels_ref = list(b["labels"]); idx = [labels_ref.index(p) for p in label_order]
        elif list(b["labels"]) != labels_ref:
            raise ValueError("seed label order mismatch")
        Ws.append(b["W"].numpy()[idx].astype(np.float64))
        for p in label_order:
            aucs[p].append(float(b["test_per_label_auc"][p]))
    Ws = np.stack(Ws, 0)                       # (S,P,D)
    Wbar = Ws.mean(0).astype(np.float32)

    npz = np.load(BANK_NPZ, allow_pickle=True, mmap_mode="r")
    emb = npz["emb"]
    concepts = [str(c) for c in np.load(BANK_NPZ, allow_pickle=True)["concepts"]]
    M, D = emb.shape
    mean = np.zeros(D, np.float64)
    for i in range(0, M, 40000):
        mean += emb[i:i + 40000].astype(np.float64).sum(0)
    mean = (mean / M).astype(np.float32)

    def cn(rows):
        v = rows.astype(np.float32) - mean
        v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
        return v

    coef_mean = np.zeros((len(label_order), M), np.float32)
    for i in range(0, M, 40000):
        coef_mean[:, i:i + 40000] = Wbar @ cn(emb[i:i + 40000]).T
    top_idx = {pi: np.argsort(-coef_mean[pi])[:N_TOP].tolist() for pi in range(len(label_order))}

    sel = sorted({c for v in top_idx.values() for c in v})
    ec_sel = cn(emb[sel]); pos = {c: r for r, c in enumerate(sel)}
    perseed = np.einsum("spd,nd->spn", Ws.astype(np.float32), ec_sel)   # (S,P,n_sel)

    naive = {}
    for pi, ph in enumerate(label_order):
        rows = []
        for ci in top_idx[pi]:
            v = perseed[:, pi, pos[ci]]
            rows.append({"concept": concepts[ci], "mean": float(v.mean()),
                         "std": float(v.std(ddof=1)), "per_seed": [float(x) for x in v]})
        au = np.array(aucs[ph], np.float64)
        half = TCRIT_19 * au.std(ddof=1) / math.sqrt(len(au))
        naive[ph] = {"concepts": rows, "auc": float(au.mean()),
                     "auc_lo": float(au.mean() - half), "auc_hi": float(au.mean() + half)}
    return naive


def build_panels(recompute=False):
    refine = json.loads(REFINE.read_text())
    if CTPA:
        # high-value CTPA phenotypes only, positive refinement gain, ranked by gain
        pool = [r for r in refine if str(r["phecode"]) in CTPA_PHECODES and r["auc_gain"] > 0]
        top = sorted(pool, key=lambda z: -z["auc_gain"])[:N_PANELS]
        print(f"CTPA selection: {len(top)} phenotypes (positive-gain, clinically-clean)")
    else:
        top = sorted(refine, key=lambda z: -z["auc_gain"])[:N_PANELS]
    label_order = [r["phenotype"] for r in top]

    # Naive/red values are independent of --nosd/--ctpa (always the Adam full-bank
    # center+normalize audit), so cache by phenotype and only compute MISSING ones —
    # merging keeps both the global-top-8 and CTPA phenotypes warm across runs.
    naive = json.loads(CACHE.read_text()) if (CACHE.exists() and not recompute) else {}
    missing = [p for p in label_order if p not in naive]
    if missing:
        print(f"computing naive full-bank audit for {len(missing)} uncached phenotype(s): {missing}")
        naive.update(compute_naive(missing))
        CACHE.write_text(json.dumps(naive))

    boot_src = TRUSTED_BOOT_PERSEED if TRUSTED_BOOT_PERSEED.exists() else TRUSTED_BOOT
    trusted_boot = json.loads(boot_src.read_text())
    print(f"trusted bootstrap source: {boot_src.name}")

    panels = []
    for r in top:
        ph = r["phenotype"]
        # Candidates = all stored trusted positives (top-10 by point fit), each with
        # its 20-train-bootstrap mean/std (and per-seed values if the perseed file is
        # present). We keep/sort the top-N_TOP by the bootstrap MEAN (the s.d. error
        # bar still shows each concept's spread), matching the naive panel's ranking.
        tb = trusted_boot.get(ph, {}).get("trusted", {})
        cand = [{"concept": c["concept"],
                 "concept_index": int(c.get("concept_index", -1)),
                 "point": float(c["importance_point"]),
                 "boot_mean": float(c["importance_boot_mean"]),
                 "boot_std": float(c["importance_boot_std"]),
                 "lower": float(c["importance_boot_mean"]) - float(c["importance_boot_std"]),
                 "sign_consistency": float(c.get("sign_consistency", float("nan"))),
                 "per_seed": [float(x) for x in c.get("per_seed", [])]}
                for c in tb.get("positive", [])]
        cand.sort(key=lambda d: -d["boot_mean"])   # select top-N by bootstrap mean
        tpos = cand[:N_TOP]
        before_th, after_th = get_themes(ph, r)
        panels.append({
            "phenotype": ph,
            "naive": naive[ph]["concepts"],
            "trusted": tpos,
            "auc_before": r["auc_before"], "auc_after": r["auc_after"], "auc_gain": r["auc_gain"],
            "before_theme": before_th, "after_theme": after_th,
        })
        print(f"{ph:42s} refine {r['auc_before']:.3f}->{r['auc_after']:.3f} (+{r['auc_gain']:.3f}) | "
              f"naive top={naive[ph]['concepts'][0]['concept'][:34]!r} | "
              f"trusted top={(tpos[0]['concept'][:34] if tpos else '-')!r}")
    return panels


# ----------------------------------------------------------------------------- draw
def draw_naive(ax, panel, xmax, rng):
    rows = sorted(panel["naive"][:N_TOP], key=lambda r_: r_["mean"])
    y = np.arange(len(rows))
    means = [c["mean"] for c in rows]; stds = [c["std"] for c in rows]
    ax.set_axisbelow(True); ax.xaxis.grid(True, color="0.88", lw=1.2, zorder=0)
    ax.barh(y, means, color=RED, edgecolor="black", linewidth=1.0, height=0.62, zorder=2)
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="black", elinewidth=1.5,
                capsize=4, capthick=1.5, zorder=4)
    for yi, c in zip(y, rows):
        v = c.get("per_seed", [])
        if v:
            ax.scatter(v, yi + rng.uniform(-0.16, 0.16, len(v)), s=14,
                       facecolors="none", edgecolors="0.15", linewidths=0.8, zorder=5)
    ax.axvline(0, color="black", lw=1.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([wrap(c["concept"], 28) for c in rows], fontsize=11.5)
    for t in ax.get_yticklabels():
        t.set_color(DRED)
    ax.tick_params(axis="y", length=4, width=1.1); ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(-0.04 * xmax, xmax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Without intervention  ·  naive full-bank audit (Adam)",
                 fontsize=11.5, fontweight="bold", color=DRED, pad=6, loc="left")


def draw_trusted(ax, panel, rng):
    rows = sorted(panel["trusted"][:N_TOP], key=lambda r_: r_["boot_mean"])   # rank by mean
    y = np.arange(len(rows))
    mean = np.array([c["boot_mean"] / TRUST_SCALE for c in rows])   # 1e3 units (sd) or raw (nosd)
    std = np.array([c["boot_std"] / TRUST_SCALE for c in rows])
    ax.set_axisbelow(True); ax.xaxis.grid(True, color="0.88", lw=1.2, zorder=0)
    ax.barh(y, mean, color=BLUE, edgecolor="black", linewidth=1.0, height=0.62, zorder=2)
    ax.errorbar(mean, y, xerr=std, fmt="none", ecolor="black", elinewidth=1.5,
                capsize=4, capthick=1.5, zorder=4)
    lo_pts = []
    for yi, c in zip(y, rows):
        v = c.get("per_seed", [])
        if v:
            vv = np.asarray(v, np.float64) / TRUST_SCALE
            lo_pts.append(vv)
            ax.scatter(vv, yi + rng.uniform(-0.16, 0.16, len(vv)), s=14,
                       facecolors="none", edgecolors="0.15", linewidths=0.8, zorder=5)
    ax.axvline(0, color="black", lw=1.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([wrap(c["concept"], 28) for c in rows], fontsize=11.5)
    for t in ax.get_yticklabels():
        t.set_color(DBLUE)
    ax.tick_params(axis="y", length=4, width=1.1); ax.tick_params(axis="x", labelsize=11)
    if len(rows):
        hi_vals = [float((mean + std).max())] + [float(v.max()) for v in lo_pts]
        lo_vals = [float((mean - std).min())] + [float(v.min()) for v in lo_pts]
        hi = max(hi_vals) * 1.12
        lo = min(lo_vals)
        ax.set_xlim(min(-0.04 * hi, lo * 1.08), hi)
    else:
        ax.set_xlim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("With intervention  ·  trusted-subset audit (LBFGS, 20-seed bootstrap)"
                 + ("  ·  importance = llm_emb · w$_z$ (no z-rescale)" if NOSD else ""),
                 fontsize=11.5, fontweight="bold", color=DBLUE, pad=6, loc="left")


def draw_graph(ax, panel, asp):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    R = 0.205; ry = R * asp
    pred = (0.74, 0.5); bef = (0.20, 0.82); aft = (0.20, 0.18)

    def circle(c, txt, face, edge, tcol):
        ax.add_patch(Ellipse(c, 2 * R, 2 * ry, facecolor=face, edgecolor=edge, lw=2.0, zorder=3, clip_on=False))
        ax.text(c[0], c[1], txt, ha="center", va="center", fontsize=11.5, color=tcol, fontweight="bold", zorder=4)

    circle(bef, panel["before_theme"], "#F4DEDE", DRED, DRED)
    circle(aft, panel["after_theme"], "#DDEAF6", DBLUE, DBLUE)
    disp = DISPLAY.get(panel["phenotype"], cap(panel["phenotype"]))
    circle(pred, textwrap.fill(disp + " prediction", 15), CIRC, "black", "black")

    for (cx, cy), col, lab, dy in [(bef, DRED, "proxy signal", 0.06), (aft, DBLUE, "direct signal", -0.06)]:
        ax.annotate("", xy=(pred[0] - R + 0.01, pred[1] + (cy - pred[1]) * 0.16), xytext=(cx + R - 0.01, cy),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=col,
                                    connectionstyle="arc3,rad=%.2f" % (-0.16 if dy > 0 else 0.16)), zorder=2)
        ax.text((cx + pred[0]) / 2, (cy + pred[1]) / 2 + dy, lab, fontsize=10.5, color=col,
                fontweight="bold", ha="center", va="center")
    ax.text(0.47, 1.05, f"AUROC {panel['auc_before']:.3f} $\\rightarrow$ {panel['auc_after']:.3f}  "
            f"(+{panel['auc_gain']:.3f})", ha="center", va="bottom", fontsize=12.5, fontweight="bold")


def main():
    recompute = "--recompute" in sys.argv
    panels = build_panels(recompute=recompute)
    rng = np.random.default_rng(0)
    xmax = max(c["mean"] + c["std"] for p in panels for c in p["naive"] if c["mean"] > 0) * 1.12

    NR = N_PANELS
    FIGW, FIGH = 21, 33
    fig = plt.figure(figsize=(FIGW, FIGH))

    top_m, bot_m = 0.965, 0.015
    row_h = (top_m - bot_m) / NR
    inner_h = row_h * 0.66
    naive_x, naive_w = 0.085, 0.165
    trust_x, trust_w = 0.42, 0.165
    graph_x, graph_w = 0.745, 0.155
    graph_asp = (graph_w * FIGW) / (inner_h * FIGH)

    for k, p in enumerate(panels):
        y0 = top_m - (k + 1) * row_h + (row_h - inner_h) * 0.5
        disp = DISPLAY.get(p["phenotype"], cap(p["phenotype"]))
        fig.text(0.012, y0 + inner_h + row_h * 0.10, f"{k+1}.  {disp}",
                 fontsize=15.5, fontweight="bold", va="bottom")
        draw_naive(fig.add_axes([naive_x, y0, naive_w, inner_h]), p, xmax, rng)
        draw_trusted(fig.add_axes([trust_x, y0, trust_w, inner_h]), p, rng)
        draw_graph(fig.add_axes([graph_x, y0, graph_w, inner_h]), p, graph_asp)

    mid_imp = ("importance = llm_emb · w$_z$ (plain, no 1/sd)" if NOSD else "importance ×10³")
    main_title = ("High-value CTPA phenotypes ranked by trusted-refinement AUROC gain: concept auditing (f2llm)"
                  if CTPA else
                  "Top-8 AUROC-gain phenotypes: concept auditing with vs without concept intervention (f2llm)")
    fig.text(0.5, 0.992, main_title
             + ("  —  trusted importance = llm_emb · w$_z$ (no z-rescale)" if NOSD else ""),
             ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.978,
             "Left: naive full-bank drivers (Adam probe, 20-seed scatter).  Middle: trusted-subset drivers after "
             f"purification (LBFGS, {mid_imp}, mean ± s.d. over 20 train-bootstrap seeds; top-5 ranked by "
             "mean).  Right: naive (red) $\\rightarrow$ phenotype $\\leftarrow$ trusted (blue).",
             ha="center", va="top", fontsize=12.5, color="0.3")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
