#!/usr/bin/env python3
"""
Merlin-Fig.-style circular RADAR of per-phenotype AUROC on the INSPECT phenotypes. Each of the
two methods is a coloured polygon (line + light fill); spokes = phenotypes, grouped into
PheWAS-category SECTORS (gaps + thick black outer arcs + category labels) exactly like
Blankemeier et al.'s Merlin wheel (Image #1). Sectors/arcs/labels are black; only the two
methods carry colour (faithful to the reference).

The radar companion to plot_forest.py — the "hero overview" view. CIs are intentionally dropped
(the forest plot stays the CI figure).

Label set (--labels):
  86      : the 86 applicable/trusted phenotypes (exp4_86), read from plot_forest.py's dumped
            forest_*_data.csv.  [default]
  all221  : ALL 221 INSPECT phecodes, read straight from the f2llm result JSONs + LP seed coefs
            (includes 6 non-clinical "Other/Admin" codes the 86-set excluded). 221 spokes do NOT
            fit on one wheel — use --split 2 (or 3) to fan them across category-balanced wheels.

Radial scaling (--scale):
  per_spoke : each spoke normalised to its OWN max (Merlin-EXACT, headline/default): rim = max
              AUROC on that spoke rounded UP into {0.70,0.75,…,1.00}, rings at 0.4/0.6/0.8/1.0 x
              that max. Matches the reference; radius NOT comparable across spokes (axes whose true
              max < 0.70 are floored to 0.70, so weak phenotypes look stronger).
  shared    : ONE scale, 0.35 at centre -> 0.90 at rim, with chance (0.50) marked by a dashed
              ring. Comparable across spokes and does not clamp below-chance results. Recommended
              for performance-ranked layouts; output gets the "_shared" suffix.

Modes (--mode):
  ctclip   : Ours (f2llm) vs CT-CLIP (CT-RATE)                          (2 arms)
  zs_vs_lp : Linear probe (f2llm) vs Zero-shot (f2llm)                  (2 arms)
  tri      : CT-CLIP zero-shot · f2llm zero-shot · f2llm linear probe   (3 arms, all221 only)
  quad     : CT-CLIP vs f2llm × zero-shot vs matched linear probe       (4 arms, all221 only)

Layout order (--order):
  ctpa       : preserve the original CTPA-relevance panel priority.                  [default]
  advantage  : quad mode only; rank clinical groups by mean paired f2llm − CT-CLIP AUROC under
               the matched 20-seed linear-probe protocol. Zero-shot differences are reported as
               a secondary metric, not mixed into the rank. Other/Admin is disclosed and placed
               last. Panels are contiguous blocks of this global order, balanced by spoke count.
               Writes a *_ranking.csv audit table.

Linear-probe data (--lp-source; all221 only):
  auto       : preserve legacy behaviour: seed42 for ``ctpa`` order and perseed20 for
               ``advantage`` order.                                                   [default]
  seed42     : the original matched fairA seed-42 JSONs (exact legacy reproduction).
  perseed20  : per-phenotype means across the 20 matched fairA probe seeds. Use this with
               ``--scale per_spoke --order ctpa`` for the Merlin-style clinical layout backed by
               the more stable 20-seed estimates.

--grid : after writing the individual split wheels, also tile them (≤4) into one 2×2 figure with a
         single shared legend at the top (the Merlin multi-panel layout).

Output: forest/radar_<tag>[_all221][_<mode>][_p<i>][_shared].png  (+ ..._layout.csv);
        with --grid also forest/radar_<tag>[_all221][_<mode>]_2x2.png.
Usage:  python plot_radar.py --labels all221 --mode quad --split 4 --panel a --grid \
                              --order advantage --scale shared
        python plot_radar.py --labels all221 --mode quad --split 4 --panel a --grid \
                              --order ctpa --scale per_spoke --lp-source perseed20 \
                              --tag f2llm_lp20
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# Nature/Merlin house style: Helvetica/Arial sans-serif, pure black furniture, italic math n.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "mathtext.default": "it",
    "text.color": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})

HERE = Path(__file__).resolve().parent
INSPECT = HERE.parent
EXP1 = INSPECT.parent
EXPERIMENTS = EXP1.parent
ROOT = EXPERIMENTS.parent
LIST_86 = EXPERIMENTS / "exp4_confounder_audit" / "exp4_86_phenotypes.csv"
PHECODES = INSPECT / "phecodes.csv"
F2LLM_ZS = ROOT / "outputs/v1/external/phenotype__test__f2llm__results.json"
CTCLIP_RES = EXP1 / "ctclip/results/inspect_pheno__native__results.json"
LP_DIR = ROOT / "outputs/v1/external"
LP_GLOB = "phenotype__linear_f2llm__seed*__lr3e-2__coefs.pt"
# "fairA" linear-probe set — CT-CLIP and f2llm probed under one matched protocol
# (same train/val/test split and per-model k-fold lr selection). The seed-42 JSONs reproduce the
# original quad by default; the newer 20-seed arrays are used for --order advantage and whenever
# --lp-source perseed20 is requested, so the displayed probe curves need not depend on one
# initialization.
FAIRA = ROOT / "outputs/v1/probe_compare/fairA"
CTCLIP_LP_JSON = FAIRA / "inspect_pheno__ctclip.json"
F2LLM_LP_JSON = FAIRA / "inspect_pheno__f2llm.json"
CTCLIP_LP_20 = FAIRA / "perseed/inspect__ctclip.npz"
F2LLM_LP_20 = FAIRA / "perseed/inspect__f2llm.npz"

# ── per-mode arm list: [(model_key, display, colour), …] drawn bottom→top (last = highlighted, on
# top). For 86 the model_key matches the forest CSV `model` column; for all221 the arms are loaded
# directly from the source result JSONs / LP seed coefs (see load_records_221). ──
# Colorblind-safe palette (Okabe–Ito). Design: the f2llm pair shares ONE blue hue as a light→dark
# ramp (zero-shot → linear-probe = the hero, darkest/most prominent), while CT-CLIP — a different
# model family / the baseline — gets a contrasting warm orange.
OI_ORANGE = "#E69F00"     # CT-CLIP (baseline)
OI_GREEN = "#009E73"      # f2llm zero-shot (bluish-green)
OI_BLUE = "#0072B2"       # f2llm linear probe (dark hero)
OI_SKY = "#56B4E9"        # spare (lighter blue)
OI_VERMILLION = "#D55E00" # spare
# ── 4-arm "quad" palette: HUE encodes the model (CT-CLIP = cool blue baseline,
# f2llm = warm orange = ours); SHADE encodes the regime (zero-shot = light, linear
# probe = dark = the stronger result). So baseline-vs-ours and zero-shot-vs-probe
# are each readable at a glance, and the two darkest lines are the headline probes.
CTCLIP_ZS_C = "#6BB6DB"   # CT-CLIP zero-shot     — light blue
CTCLIP_LP_C = "#0B3D91"   # CT-CLIP linear probe  — deep blue
F2LLM_ZS_C = "#EF9B4B"    # f2llm zero-shot       — light orange
F2LLM_LP_C = "#9E2B00"    # f2llm linear probe    — dark rust (hero)
MODE_SPEC = {
    "ctclip": dict(csv="forest_f2llm_data.csv",
                   arms=[("ctclip", "CT-CLIP (CT-RATE)", OI_ORANGE),
                         ("ours_f2llm", "Ours (f2llm)", OI_BLUE)]),
    "zs_vs_lp": dict(csv="forest_f2llm_zs_vs_lp_data.csv",
                     arms=[("zs", "Zero-shot (f2llm)", OI_SKY),
                           ("lp", "Linear probe (f2llm)", OI_BLUE)]),
    # 3-arm headline (all221 only): CT-CLIP zero-shot vs f2llm zero-shot vs f2llm linear probe.
    "tri": dict(arms=[("ctclip", "CT-CLIP (zero-shot)", OI_ORANGE),
                      ("f2llm_zs", "f2llm (zero-shot)", OI_GREEN),
                      ("f2llm_lp", "f2llm (linear probe)", OI_BLUE)]),
    # 4-arm 2×2 (all221 only): CT-CLIP vs f2llm × zero-shot vs linear probe. Both linear
    # probes come from the fairA matched protocol (see FAIRA) so the comparison is fair.
    # arms drawn bottom→top: the f2llm linear probe (hero) is last, hence on top.
    "quad": dict(arms=[("ctclip", "CT-CLIP (zero-shot)", CTCLIP_ZS_C),
                       ("ctclip_lp", "CT-CLIP (linear probe)", CTCLIP_LP_C),
                       ("f2llm_zs", "ACT (zero-shot)", F2LLM_ZS_C),
                       ("f2llm_lp_fair", "ACT (linear probe)", F2LLM_LP_C)]),
}

# ── short display names for long clinical names (keeps spokes from over-running the arc). Full
# names are kept in the layout CSV. Anything not listed is shown IN FULL and wrapped (no ellipsis). ──
ABBREV = {
    "Cancer within the respiratory system": "Respiratory-system cancer",
    "Malignant neoplasm of female breast": "Malig. neoplasm, female breast",
    "Secondary malignancy of lymph nodes": "Secondary malig. (lymph node)",
    "Secondary malignancy of respiratory organs": "Secondary malig. (respiratory)",
    "Secondary malignancy of bone": "Secondary malig. (bone)",
    "Type 2 diabetes with renal manifestations": "T2D w/ renal manifestations",
    "Overweight, obesity and other hyperalimentation": "Overweight / hyperalimentation",
    "Anemia in chronic kidney disease": "Anemia in CKD",
    "Rheumatic disease of the heart valves": "Rheumatic heart-valve disease",
    "Hypertensive heart and/or renal disease": "Hypertensive heart/renal dis.",
    "Hypertensive chronic kidney disease": "Hypertensive CKD",
    "Pulmonary embolism and infarction, acute": "Acute PE / infarction",
    "Chronic pulmonary heart disease": "Chronic pulmonary heart dis.",
    "Acute pulmonary heart disease": "Acute pulmonary heart dis.",
    "Congestive heart failure (CHF) NOS": "CHF NOS",
    "Heart failure with reduced EF [Systolic or combined heart failure]": "HF, reduced EF (systolic)",
    "Heart failure with preserved EF [Diastolic heart failure]": "HF, preserved EF (diastolic)",
    "Other venous embolism and thrombosis": "Other venous thromboembolism",
    "Deep vein thrombosis [DVT]": "Deep vein thrombosis (DVT)",
    "Pneumonitis due to inhalation of food or vomitus": "Aspiration pneumonitis",
    "Pulmonary congestion and hypostasis": "Pulmonary congestion",
    "Other pulmonary inflamation or edema": "Other pulmonary inflam./edema",
    "Pulmonary collapse; interstitial and compensatory emphysema": "Pulmonary collapse / emphysema",
    "Other disorders of peritoneum": "Other peritoneal disorders",
    "Other disorders of intestine": "Other intestinal disorders",
    "Other chronic nonalcoholic liver disease": "Other chronic NAFLD",
    "Cirrhosis of liver without mention of alcohol": "Cirrhosis (non-alcoholic)",
    "Chronic renal failure [CKD]": "Chronic renal failure (CKD)",
    "Chronic Kidney Disease, Stage III": "CKD, Stage III",
    "Chronic kidney disease, Stage I or II": "CKD, Stage I–II",
    "Superficial cellulitis and abscess": "Superficial cellulitis/abscess",
    "Nonspecific abnormal findings on radiological and other examination of musculoskeletal system":
        "Nonspecific MSK radiol. findings",
    "Systemic inflammatory response syndrome (SIRS)": "SIRS",
    # all221-only non-clinical / long codes
    "Persons with potential health hazards related to socioeconomic, psychosocial, and other circumstances":
        "Psychosocial health hazards",
    "Other ill-defined and unknown causes of morbidity and mortality": "Ill-defined causes of M&M",
    "Complications of surgical and medical procedures": "Surgical/medical complications",
    "Vertiginous syndromes and other disorders of vestibular system": "Vestibular disorders",
    "Other diseases of respiratory system, not elsewhere classified": "Other respiratory disease NEC",
    # long names that would otherwise overrun the arc — abbreviated in full (no ellipsis)
    "Nonspecific elevation of levels of transaminase or lactic acid dehydrogenase [LDH]":
        "Elevated transaminase / LDH",
    "Toxic effect of (non-ethyl) alcohol and petroleum and other solvents":
        "Toxic non-ethanol alcohol/solvents",
    "Iron deficiency anemias, unspecified or not due to blood loss": "Iron-deficiency anemia, unspec.",
    "Dependence on respirator [Ventilator] or supplemental oxygen": "Ventilator / oxygen dependence",
    "Delirium dementia and amnestic and other cognitive disorders": "Dementia / amnestic disorders",
    "Dizziness and giddiness (Light-headedness and vertigo)": "Dizziness / light-headedness",
    "Complications of transplants and reattached limbs": "Transplant / reattachment compl.",
    "Other diseases of blood and blood-forming organs": "Other blood / marrow diseases",
    "Delirium due to conditions classified elsewhere": "Delirium from other conditions",
    "Other symptoms/disorders or the urinary system": "Other urinary symptoms/disorders",
    "Nonspecific findings on examination of blood": "Nonspecific blood findings",
    "Cardiac and circulatory congenital anomalies": "Congenital cardiac anomalies",
    "Other symptoms involving abdomen and pelvis": "Other abdomen/pelvis symptoms",
    "Musculoskeletal symptoms referable to limbs": "Musculoskeletal limb symptoms",
    "Disorders of calcium/phosphorus metabolism": "Calcium/phosphorus disorders",
    "Other disorders of the kidney and ureters": "Other kidney/ureter disorders",
    "Epilepsy, recurrent seizures, convulsions": "Epilepsy / recurrent seizures",
    "Abnormal electrocardiogram [ECG] [EKG]": "Abnormal ECG/EKG",
    "Paroxysmal supraventricular tachycardia": "Paroxysmal SVT",
    "Paroxysmal tachycardia, unspecified": "Paroxysmal tachycardia NOS",
    "Purpura and other hemorrhagic conditions": "Purpura / hemorrhagic conditions",
}


def _display(name: str) -> str:
    return ABBREV.get(name, name)            # full name (or a clean abbreviation) — never ellipsized


# ── PheWAS-category sectors, derived from the phecode (auto + reproducible) ───────────
CATEGORY_OVERRIDE = {771.1: "Symptoms", 1013.0: "Symptoms",
                     994.1: "Sepsis/SIRS", 994.2: "Sepsis/SIRS", 994.21: "Sepsis/SIRS"}
# thematic order around the wheel; a contiguous split (--split) cuts this list into balanced groups
CATEGORY_ORDER = [
    "Infectious", "Neoplasms", "Hematopoietic", "Endocrine/Metabolic", "Circulatory",
    "Sepsis/SIRS", "Genitourinary", "Respiratory", "Digestive", "Mental", "Neurological/Sense",
    "Musculoskeletal", "Dermatologic", "Congenital", "Pregnancy", "Symptoms", "Injury", "Other/Admin",
]
CATEGORY_SHORT = {
    "Endocrine/Metabolic": "Endocrine /\nMetabolic", "Genitourinary": "Genito-\nurinary",
    "Musculoskeletal": "Musculo-\nskeletal", "Dermatologic": "Derm.", "Hematopoietic": "Hemato-\npoietic",
    "Sepsis/SIRS": "Sepsis /\nSIRS", "Neurological/Sense": "Neuro /\nSense", "Other/Admin": "Other /\nAdmin",
}


def phecode_category(phecode: str) -> str:
    x = float(phecode)
    if round(x, 2) in CATEGORY_OVERRIDE:
        return CATEGORY_OVERRIDE[round(x, 2)]
    for lo, hi, name in [
        (1, 140, "Infectious"), (140, 240, "Neoplasms"), (240, 280, "Endocrine/Metabolic"),
        (280, 290, "Hematopoietic"), (290, 320, "Mental"), (320, 390, "Neurological/Sense"),
        (390, 460, "Circulatory"), (460, 520, "Respiratory"), (520, 580, "Digestive"),
        (580, 630, "Genitourinary"), (630, 680, "Pregnancy"), (680, 710, "Dermatologic"),
        (710, 760, "Musculoskeletal"), (760, 780, "Congenital"), (780, 800, "Symptoms"),
        (800, 1000, "Injury"),
    ]:
        if lo <= x < hi:
            return name
    return "Other/Admin"                      # 1000+ custom administrative codes


# ── geometry / style constants ───────────────────────────────────────────────────────
R_OUT = 1.0
A_MIN, A_MAX = 0.35, 0.90
A_FLOOR = A_MIN
RINGS_SHARED = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RINGS_PS = [0.4, 0.6, 0.8, 1.0]
LADDER_MAX = 58              # full per-spoke tick ladder only ≤ this many spokes/wheel (else numbers
                            # shrink below ~5pt and collide near the centre); denser wheels = rim-only
GAP_DEG = 3.2
START_DEG = 90.0
ARC_DR = 0.82           # sector arc radius beyond the rim — kept OUTSIDE the spoke labels so the
CAT_DR = 1.02           # thick black arc never crosses the (long) phenotype names
LABEL_DR = 0.05
WRAP_MIN = 20

SPOKE_FS = 13.0          # base spoke-label size (86 wheel); auto-scaled down on denser split wheels
CAT_FS = 25.0            # outer PheWAS-group labels — large (scaled ×fs_mul again in the 2×2 grid)
TICK_FS = 14.0
LEGEND_FS = 22.0
# The 2x2 manuscript composite applies a 1.25x scale below, making these
# panel labels prominent after the figure is fit to manuscript text width.
PANEL_FS = 34.0
LINE_LW = 2.8
FILL_ALPHA = 0.15
MARGIN = 0.05            # blank ring beyond the category labels (R_OUT + CAT_DR + MARGIN = axis limit)


def _auc_to_r_shared(a: float) -> float:
    a = A_FLOOR if not np.isfinite(a) else min(max(a, A_FLOOR), A_MAX)
    return (a - A_MIN) / (A_MAX - A_MIN) * R_OUT


def _wrap2(text: str, width: int = WRAP_MIN) -> str:
    """Balance into up to 3 lines, MINIMISING the longest line (so names never overrun the arc).
    Prefer 2 lines; fall back to 3 only when 2 would leave a line longer than ~width+1 chars."""
    words = text.split()
    if len(text) <= width or len(words) < 2:
        return text

    def best(n):                                         # n contiguous groups, min the longest line
        n = min(n, len(words))
        out, cost = None, None
        for cuts in itertools.combinations(range(1, len(words)), n - 1):
            bnds = [0, *cuts, len(words)]
            lines = [" ".join(words[bnds[i]:bnds[i + 1]]) for i in range(n)]
            c = max(len(s) for s in lines)
            if cost is None or c < cost:
                cost, out = c, lines
        return out, cost

    for n in (2, 3):
        lines, cost = best(n)
        if cost <= width + 1 or n == 3:
            return "\n".join(lines)
    return "\n".join(lines)


def _pla(d: dict) -> dict:
    """results.json -> {clean label: auc}."""
    p = d["per_label_auc"]
    p = p if isinstance(p, dict) else dict(zip(d["labels"], p))
    return {str(k).strip(): float(v) for k, v in p.items()}


def _records(names, name2pc, arm_plas: dict) -> pd.DataFrame:
    """arm_plas: {arm_key: {clean name: auc}} -> df with one auc_<key> column per arm."""
    rows = []
    for nm in names:
        pc = name2pc.get(nm)
        if pc is None:
            continue
        row = dict(phenotype=nm, display=_display(nm), phecode=str(pc),
                   category=phecode_category(str(pc)))
        for k, pla in arm_plas.items():
            row[f"auc_{k}"] = pla.get(nm, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def load_records_86(mode: str):
    spec = MODE_SPEC[mode]
    if "csv" not in spec:
        raise SystemExit(f"mode={mode!r} is all221-only — pass --labels all221")
    df = pd.read_csv(HERE / spec["csv"]); df["phenotype"] = df["phenotype"].str.strip()
    auc = df.pivot_table(index="phenotype", columns="model", values="auc", aggfunc="first")
    meta = pd.read_csv(LIST_86); meta["phenotype"] = meta["phenotype"].str.strip()
    name2pc = {n: str(p) for n, p in zip(meta["phenotype"], meta["phecode"])}
    names = meta["phenotype"].tolist()
    arm_plas = {k: {n: float(auc.loc[n, k]) for n in names if n in auc.index and k in auc.columns}
                for (k, _d, _c) in spec["arms"]}
    return _records(names, name2pc, arm_plas), spec["arms"]


def _lp_mean_pla() -> dict:
    """f2llm linear-probe AUROC averaged over the 20 seed coef files."""
    import torch
    files = sorted(LP_DIR.glob(LP_GLOB))
    if not files:
        raise SystemExit(f"no LP coefs match {LP_DIR}/{LP_GLOB}")
    acc: dict[str, list] = {}
    for f in files:
        c = torch.load(f, map_location="cpu"); pla = c["test_per_label_auc"]
        pla = pla if isinstance(pla, dict) else dict(zip(c["labels"], pla))
        for k, v in pla.items():
            acc.setdefault(str(k).strip(), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _faira_perseed_mean_pla(path: Path) -> dict:
    """Matched fairA per-label AUROC averaged across the saved 20 probe seeds."""
    if not path.exists():
        raise SystemExit(f"missing matched 20-seed linear-probe data: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"labels", "seeds", "per_label_auc"}
        if not required.issubset(z.files):
            raise SystemExit(f"{path} lacks required arrays: {sorted(required - set(z.files))}")
        labels = [str(x).strip() for x in z["labels"]]
        seeds = z["seeds"]
        auc = z["per_label_auc"]
    if auc.ndim != 2 or auc.shape != (len(seeds), len(labels)):
        raise SystemExit(f"unexpected per_label_auc shape in {path}: {auc.shape}")
    if len(seeds) != 20:
        raise SystemExit(f"expected 20 matched probe seeds in {path}, found {len(seeds)}")
    return dict(zip(labels, auc.mean(axis=0).astype(float)))


def load_records_221(mode: str, lp_source: str = "seed42"):
    spec = MODE_SPEC[mode]
    zs = _pla(json.loads(F2LLM_ZS.read_text()))
    names = [str(l).strip() for l in json.loads(F2LLM_ZS.read_text())["labels"]]
    phe = pd.read_csv(PHECODES)
    name2pc = {str(s).strip(): str(p) for s, p in zip(phe["phecode_str"], phe["phecode"])}
    # map every arm key to its per-label-AUROC source (lazy for the heavy LP load)
    if lp_source not in {"seed42", "perseed20"}:
        raise ValueError(f"unknown lp_source={lp_source!r}")
    ctclip_lp = (lambda: _faira_perseed_mean_pla(CTCLIP_LP_20)) if lp_source == "perseed20" else (
        lambda: _pla(json.loads(CTCLIP_LP_JSON.read_text())))
    f2llm_lp = (lambda: _faira_perseed_mean_pla(F2LLM_LP_20)) if lp_source == "perseed20" else (
        lambda: _pla(json.loads(F2LLM_LP_JSON.read_text())))
    src = {"zs": lambda: zs, "ours_f2llm": lambda: zs, "f2llm_zs": lambda: zs,
           "ctclip": lambda: _pla(json.loads(CTCLIP_RES.read_text())),
           "lp": _lp_mean_pla, "f2llm_lp": _lp_mean_pla,
           # fairA matched-protocol linear probes (quad mode)
           "ctclip_lp": ctclip_lp, "f2llm_lp_fair": f2llm_lp}
    cache: dict = {}
    arm_plas = {}
    for (k, _d, _c) in spec["arms"]:
        if k not in cache:
            cache[k] = src[k]()
        arm_plas[k] = cache[k]
    return _records(names, name2pc, arm_plas), spec["arms"]


def _finalize(df: pd.DataFrame, arm_keys) -> pd.DataFrame:
    """Keep mappable rows, sort by (category order, phecode), add the per-spoke max over all arms."""
    df = df[df["category"].isin(CATEGORY_ORDER)].copy()
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df["_co"] = df["category"].map(order); df["_pc"] = df["phecode"].astype(float)
    df = df.sort_values(["_co", "_pc"]).drop(columns=["_co", "_pc"]).reset_index(drop=True)
    mx = df[[f"auc_{k}" for k in arm_keys]].max(axis=1)
    df["spoke_max"] = np.clip(np.round(np.ceil(np.round(mx / 0.05, 6)) * 0.05, 2), 0.70, 1.00)
    return df


# CTPA relevance: panels a→d should lead with the groups most relevant to a PE/CTPA chest study
# (cardiovascular + pulmonary first; non-imaging groups like Mental/Dermatologic last).
CTPA_ORDER = [
    "Circulatory", "Respiratory", "Symptoms", "Sepsis/SIRS", "Neoplasms", "Hematopoietic",
    "Digestive", "Genitourinary", "Endocrine/Metabolic", "Musculoskeletal", "Neurological/Sense",
    "Injury", "Mental", "Dermatologic", "Congenital", "Infectious", "Pregnancy", "Other/Admin",
]


def _split_groups(df: pd.DataFrame, n: int, priority):
    """Partition categories into n balanced wheels, keeping every category whole.

    Categories are visited in ``priority`` order and first-fit into the earliest wheel with room
    under the balance cap. This reproduces the original CTPA-priority layout.
    """
    sizes = df["category"].value_counts().to_dict()
    rel = {c: i for i, c in enumerate(priority)}
    total = sum(sizes.values())
    cap = max(-(-total // n), max(sizes.values()))      # ceil(total/n), but never below the biggest category
    bins, loads = [[] for _ in range(n)], [0] * n
    for c in sorted(sizes, key=lambda c: rel.get(c, 99)):
        bi = next((k for k in range(n) if loads[k] + sizes[c] <= cap),
                  min(range(n), key=lambda k: loads[k]))
        bins[bi].append(c); loads[bi] += sizes[c]
    return [b for b in bins if b]                       # bin 0 → panel a → most relevant


def _split_contiguous_groups(df: pd.DataFrame, n: int, priority):
    """Split a global category ranking into balanced contiguous panels.

    Keeping each panel contiguous makes panel a contain the strongest groups, panel b the next
    strongest, and so on. We choose the cuts that minimize the worst deviation from equal spoke
    count, then total squared deviation as a deterministic tie-breaker.
    """
    sizes = df["category"].value_counts().to_dict()
    cats = [c for c in priority if c in sizes]
    if n <= 1:
        return [cats]
    if n > len(cats):
        raise SystemExit(f"cannot split {len(cats)} categories into {n} non-empty panels")
    target = sum(sizes.values()) / n
    best = None
    for cuts in itertools.combinations(range(1, len(cats)), n - 1):
        bounds = (0, *cuts, len(cats))
        blocks = [cats[bounds[i]:bounds[i + 1]] for i in range(n)]
        loads = [sum(sizes[c] for c in block) for block in blocks]
        score = (max(abs(x - target) for x in loads),
                 sum((x - target) ** 2 for x in loads), cuts)
        if best is None or score < best[0]:
            best = (score, blocks)
    return best[1]


def _add_advantage_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Add the paired f2llm-vs-CT-CLIP deltas used by the performance-ranked quad layout."""
    if mode != "quad":
        raise SystemExit("--order advantage currently requires --mode quad")
    required = ["auc_ctclip", "auc_ctclip_lp", "auc_f2llm_zs", "auc_f2llm_lp_fair"]
    missing = [c for c in required if c not in df]
    if missing:
        raise SystemExit(f"cannot rank by f2llm advantage; missing columns: {missing}")
    bad = df[required].isna().any(axis=1)
    if bad.any():
        raise SystemExit(f"cannot rank by f2llm advantage; {int(bad.sum())} rows have missing AUROC")
    df = df.copy()
    df["delta_zero_shot"] = df["auc_f2llm_zs"] - df["auc_ctclip"]
    df["delta_linear_probe"] = df["auc_f2llm_lp_fair"] - df["auc_ctclip_lp"]
    # The matched 20-seed linear probe is the single prespecified ranking statistic. Zero-shot is
    # retained beside it as a secondary metric rather than mixed in with an arbitrary weight.
    df["advantage_score"] = df["delta_linear_probe"]
    df["both_mode_win"] = (df["delta_zero_shot"] > 0) & (df["delta_linear_probe"] > 0)
    return df


def _advantage_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per group, ranked by the matched 20-seed linear-probe delta."""
    out = (df.groupby("category", as_index=False)
           .agg(n_phenotypes=("phenotype", "size"),
                mean_auc_ctclip_zero_shot=("auc_ctclip", "mean"),
                mean_auc_f2llm_zero_shot=("auc_f2llm_zs", "mean"),
                mean_delta_zero_shot=("delta_zero_shot", "mean"),
                zero_shot_wins=("delta_zero_shot", lambda x: int((x > 0).sum())),
                zero_shot_win_rate=("delta_zero_shot", lambda x: float((x > 0).mean())),
                mean_auc_ctclip_linear_probe=("auc_ctclip_lp", "mean"),
                mean_auc_f2llm_linear_probe=("auc_f2llm_lp_fair", "mean"),
                mean_delta_linear_probe=("delta_linear_probe", "mean"),
                linear_probe_wins=("delta_linear_probe", lambda x: int((x > 0).sum())),
                linear_probe_win_rate=("delta_linear_probe", lambda x: float((x > 0).mean())),
                both_mode_wins=("both_mode_win", "sum"),
                both_mode_win_rate=("both_mode_win", "mean")))
    out = out.sort_values(["mean_delta_linear_probe", "linear_probe_win_rate", "category"],
                          ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "numeric_rank", np.arange(1, len(out) + 1))
    # Other/Admin contains six nonclinical endpoints. Keep its numeric rank for auditability but
    # place it last in the clinical display so panel a is not led by administrative outcomes.
    out["placement_note"] = np.where(out["category"].eq("Other/Admin"),
                                     "placed last by design (nonclinical)", "")
    out = pd.concat([out[~out["category"].eq("Other/Admin")],
                     out[out["category"].eq("Other/Admin")]], ignore_index=True)
    out.insert(0, "group_rank", np.arange(1, len(out) + 1))
    return out


def _sort_categories(df: pd.DataFrame, categories) -> pd.DataFrame:
    """Order sectors by an explicit category list and keep phecodes ordered inside each sector."""
    rank = {c: i for i, c in enumerate(categories)}
    out = df.copy()
    out["_category_rank"] = out["category"].map(rank)
    out["_phecode_numeric"] = out["phecode"].astype(float)
    return (out.sort_values(["_category_rank", "_phecode_numeric"])
            .drop(columns=["_category_rank", "_phecode_numeric"]).reset_index(drop=True))


def _assign_angles(df: pd.DataFrame):
    """Add an 'angle' (deg) column walking clockwise from the top with inter-sector gaps; return
    (df, sector_span) where sector_span[cat] = [start_deg, end_deg]."""
    df = df.reset_index(drop=True)
    n, n_sec = len(df), df["category"].nunique()
    step = (360.0 - n_sec * GAP_DEG) / n
    angles, span, ang, prev = [], {}, START_DEG, None
    for cat in df["category"]:
        if cat != prev:
            if prev is not None:
                ang -= GAP_DEG
            span[cat] = [ang, ang]; prev = cat
        angles.append(ang - step / 2.0)
        ang -= step
        span[cat][1] = ang
    df = df.copy(); df["angle"] = angles
    return df, span


def _render_wheel(ax, lay, arms, span, scale, spoke_fs, panel="", fs_mul=1.0, ring_mul=1.0,
                  spoke_mul=None):
    """Draw one radar wheel onto `ax` (limits/furniture/arms/labels/sector arcs). Returns the
    legend handles (one per arm) so the caller decides where the legend goes. `fs_mul` scales the
    category/panel/tick furniture; `spoke_mul` (defaults to fs_mul) the rim phenotype labels;
    `ring_mul` the per-spoke ring ladder (each ring then graded by radius so the outer digits are
    large and the innermost stay small enough not to collide at the hub)."""
    per_spoke = scale == "per_spoke"
    spoke_mul = fs_mul if spoke_mul is None else spoke_mul
    ps_fs = max(4.5, spoke_fs * 0.62) * fs_mul
    ang = np.deg2rad(lay["angle"].to_numpy())
    ax.set_aspect("equal"); ax.axis("off")
    LIM = R_OUT + CAT_DR + MARGIN
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)

    def r_of(col):
        if per_spoke:
            return np.clip(lay[col].to_numpy() / lay["spoke_max"].to_numpy(), 0, 1) * R_OUT
        return lay[col].map(_auc_to_r_shared).to_numpy()

    th = np.linspace(0, 2 * np.pi, 400)
    # concentric AUROC rings drawn as clean, clearly-visible grey circles (Merlin look): darker/
    # thicker than the faint radial spokes and lifted just above the light polygon fills so the
    # inner circle reads through them (the innermost is kept digit-free below → a clean reference ring)
    for r in ([f * R_OUT for f in RINGS_PS] if per_spoke else [_auc_to_r_shared(a) for a in RINGS_SHARED]):
        ax.plot(r * np.cos(th), r * np.sin(th), color="0.6", lw=1.1, zorder=2.5)
    if not per_spoke:
        ax.plot(_auc_to_r_shared(0.5) * np.cos(th), _auc_to_r_shared(0.5) * np.sin(th),
                color="0.6", lw=0.9, dashes=(4, 3), zorder=0)
    for a in ang:
        ax.plot([0, R_OUT * np.cos(a)], [0, R_OUT * np.sin(a)], color="0.86", lw=0.5, zorder=0)

    if per_spoke and len(lay) <= LADDER_MAX:
        # Merlin-exact: label EVERY ring on EVERY spoke (0.4/0.6/0.8/1.0 × that spoke's max), as
        # small plain black numbers (thin white halo, no box — like the reference) at each gridline
        # crossing. Font shrinks with spoke count so the inner ring doesn't collide near the centre.
        # tangential room per spoke ∝ radius, so grade each ring's font by its fraction: the outer
        # ring (×max) gets the full size, the innermost (0.4×) shrinks to a floor so it doesn't collide.
        ring_out = float(np.clip(round(560.0 / len(lay), 1), 7.0, 12.0)) * ring_mul
        halo = [pe.withStroke(linewidth=1.4, foreground="white")]
        for a_deg, m in zip(lay["angle"], lay["spoke_max"]):
            a = np.deg2rad(a_deg)
            for frac in RINGS_PS:
                if frac == RINGS_PS[0]:        # innermost ring: clean circle only, no digits
                    continue                   # (the 0.4×max numbers collide into a mess at the hub)
                r = frac * R_OUT
                ax.text(r * np.cos(a), r * np.sin(a), f"{frac * m:.2f}",
                        fontsize=max(ring_out * frac, 4.6), ha="center", va="center", zorder=6,
                        color="black", path_effects=halo)
    elif per_spoke:                                  # dense wheel: rim max only (ladder would collide)
        for a_deg, m in zip(lay["angle"], lay["spoke_max"]):
            a = np.deg2rad(a_deg)
            ax.text(R_OUT * np.cos(a), R_OUT * np.sin(a), f"{m:.2f}", fontsize=ps_fs,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    else:
        for a in RINGS_SHARED:
            r = _auc_to_r_shared(a)
            ax.text(0.012, r, f"{a:.1f}", fontsize=TICK_FS * fs_mul, ha="left", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    for zi, (key, disp, color) in enumerate(arms):     # bottom→top; last arm sits on top
        r = r_of(f"auc_{key}")
        x, y = r * np.cos(ang), r * np.sin(ang)
        xc, yc = np.r_[x, x[:1]], np.r_[y, y[:1]]
        zo = 3 + zi
        if "zero-shot" not in disp.lower():            # zero-shot arms drawn line-only (no fill)
            ax.fill(xc, yc, color=color, alpha=FILL_ALPHA, zorder=zo, lw=0)
        ax.plot(xc, yc, color=color, lw=LINE_LW, zorder=zo + 0.1,
                solid_joinstyle="round", solid_capstyle="round")
        ax.plot(x, y, "o", color=color, ms=max(1.8, spoke_fs * 0.27), zorder=zo + 0.2)

    for a_deg, nm in zip(lay["angle"], lay["display"]):
        a = np.deg2rad(a_deg)
        ax.plot([R_OUT * np.cos(a), (R_OUT + 0.012) * np.cos(a)],
                [R_OUT * np.sin(a), (R_OUT + 0.012) * np.sin(a)], color="0.4", lw=0.7, zorder=4)
        deg = a_deg % 360
        rot, ha = (a_deg + 180, "right") if 90 < deg < 270 else (a_deg, "left")
        ax.text((R_OUT + LABEL_DR) * np.cos(a), (R_OUT + LABEL_DR) * np.sin(a), _wrap2(nm),
                fontsize=spoke_fs * spoke_mul, rotation=rot, rotation_mode="anchor", ha=ha,
                va="center", linespacing=0.9)

    for cat, (s_deg, e_deg) in span.items():
        inset = min(1.2, abs(s_deg - e_deg) * 0.12)
        r_arc = R_OUT + ARC_DR
        ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, angle=0,
                         theta1=e_deg + inset, theta2=s_deg - inset, lw=2.6, color="black", zorder=5))
        mid = np.deg2rad((s_deg + e_deg) / 2.0)
        deg = np.rad2deg(mid) % 360
        rot = (deg - 90) if deg <= 180 else (deg + 90)
        ax.text((R_OUT + CAT_DR) * np.cos(mid), (R_OUT + CAT_DR) * np.sin(mid),
                CATEGORY_SHORT.get(cat, cat), fontsize=CAT_FS * fs_mul, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", linespacing=0.9)

    if panel:
        ax.text(0.035, 0.975, panel, transform=ax.transAxes, fontsize=PANEL_FS * fs_mul,
                fontweight="bold", fontfamily="Arial", ha="left", va="top")

    return [Line2D([0], [0], color=c, lw=3.2, label=d) for (_k, d, c) in arms]


def _note_text(scale, note_extra=""):
    note = ("each axis scaled to its own max (rings 0.4–1.0×max)" if scale == "per_spoke"
            else "shared AUROC: 0.35 to 0.90; dashed 0.50 = chance")
    return f"{note}    ·    {note_extra}" if note_extra else note


def _draw_wheel(lay, arms, span, scale, panel, out, spoke_fs, note_extra=""):
    fig = plt.figure(figsize=(19, 19))
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    handles = _render_wheel(ax, lay, arms, span, scale, spoke_fs, panel=panel)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, -0.02),
              frameon=False, fontsize=LEGEND_FS, handlelength=1.6, borderaxespad=0.0)
    fig.text(0.5, 0.045, _note_text(scale, note_extra), fontsize=12, ha="center", va="bottom",
             color="0.35")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine_grid(parts, arms, scale, out, note="", dpi=200):
    """Tile the per-part wheels into a 2×2 figure with ONE shared legend up top (Merlin multi-panel
    layout). `parts` = [(lay, span, spoke_fs, panel_letter), …] (≤4 used). Axes are square and packed
    edge-to-edge (only a thin gutter between wheels) so each wheel fills its quadrant; fonts are
    enlarged via fs_mul, and the figure is cropped tight on save to kill the outer margin."""
    W, H = 30.0, 31.0
    # Reserve a narrow top band for the shared legend; within the 2×2 body the wheels remain
    # tightly packed with only a hair of gutter.
    POS = [[0.001, 0.492, 0.498, 0.486], [0.501, 0.492, 0.498, 0.486],   # a (TL)  b (TR)
           [0.001, 0.004, 0.498, 0.486], [0.501, 0.004, 0.498, 0.486]]   # c (BL)  d (BR)
    fig = plt.figure(figsize=(W, H))
    handles = None
    for (lay, span, fs, pnl), pos in zip(parts[:4], POS):
        ax = fig.add_axes(pos)
        handles = _render_wheel(ax, lay, arms, span, scale, fs, panel=pnl,
                                fs_mul=1.25, spoke_mul=1.32, ring_mul=1.0)
    # Lift the legend clear of two-line sector labels near 12 o'clock (notably Sepsis/SIRS).
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.022),
               ncol=len(arms), frameon=False, fontsize=LEGEND_FS * 1.55, handlelength=1.8,
               columnspacing=2.6, borderaxespad=0.0)
    if note:
        fig.text(0.5, 0.004, note, fontsize=18, ha="center", va="bottom", color="0.35")
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _outname(tag, mode, labels, scale, part, order="ctpa"):
    base = f"radar_{tag}"
    if labels == "all221":
        base += "_all221"
    if mode != "ctclip":
        base += f"_{mode}"
    if order != "ctpa":
        base += f"_{order}"
    if part is not None:
        base += f"_p{part}"
    if scale != "per_spoke":
        base += "_shared"
    return HERE / f"{base}.png"


def plot(mode="ctclip", scale="per_spoke", labels="86", split=1, tag="f2llm", panel="",
         dpi=300, grid=False, order="ctpa", lp_source="auto"):
    if labels == "all221":
        resolved_lp_source = ("perseed20" if order == "advantage" else "seed42") \
            if lp_source == "auto" else lp_source
        if order == "advantage" and resolved_lp_source != "perseed20":
            raise SystemExit("--order advantage requires --lp-source auto or perseed20")
        df, arms = load_records_221(mode, lp_source=resolved_lp_source)
    else:
        if lp_source != "auto":
            raise SystemExit("--lp-source applies only to --labels all221")
        resolved_lp_source = "legacy-86"
        df, arms = load_records_86(mode)
    keys = [k for (k, _d, _c) in arms]
    df = _finalize(df, keys)

    order_note = ""
    ranking = None
    if order == "advantage":
        df = _add_advantage_columns(df, mode)
        ranking = _advantage_group_summary(df)
        priority = ranking["category"].tolist()
        parts = _split_contiguous_groups(df, split, priority)
        order_note = ("sectors ranked by mean f2llm - CT-CLIP AUROC "
                      "(matched 20-seed linear probe); Other/Admin last")

        cat2panel = {cat: i + 1 for i, cats in enumerate(parts) for cat in cats}
        panel_stats = {}
        for i, cats in enumerate(parts, start=1):
            rows = df[df["category"].isin(cats)]
            panel_stats[i] = dict(
                panel_n_phenotypes=len(rows),
                panel_mean_delta_zero_shot=rows["delta_zero_shot"].mean(),
                panel_mean_delta_linear_probe=rows["delta_linear_probe"].mean(),
                panel_mean_advantage=rows["advantage_score"].mean(),
                panel_advantage_win_rate=float((rows["advantage_score"] > 0).mean()),
                panel_both_mode_wins=int(rows["both_mode_win"].sum()),
                panel_both_mode_win_rate=float(rows["both_mode_win"].mean()),
            )
        ranking["panel_rank"] = ranking["category"].map(cat2panel)
        ranking["panel"] = ranking["panel_rank"].map(lambda x: chr(ord("a") + int(x) - 1))
        ranking["ranking_basis"] = "mean paired AUROC delta, matched 20-seed linear probe"
        ranking["linear_probe_seeds"] = 20
        for col in next(iter(panel_stats.values())):
            ranking[col] = ranking["panel_rank"].map(lambda x, c=col: panel_stats[int(x)][c])
    else:
        parts = (_split_groups(df, split, CTPA_ORDER) if split > 1
                 else [list(dict.fromkeys(df["category"]))])

    if ranking is not None:
        rank_out = Path(str(_outname(tag, mode, labels, scale, None, order)).replace(
            ".png", "_ranking.csv"))
        ranking.to_csv(rank_out, index=False)
        print(f"wrote {rank_out.name}  ({len(ranking)} ranked groups)")

    parts_data = []
    for i, cats in enumerate(parts):
        sub = df[df["category"].isin(cats)].reset_index(drop=True)
        if order == "advantage":
            sub = _sort_categories(sub, cats)
        lay, span = _assign_angles(sub)
        # scale the spoke font to spoke density (room per spoke ∝ 1/n); 86 -> 13pt baseline,
        # up to 13.5pt on sparser wheels (e.g. ~74 spokes) where there's tangential room.
        fs = float(np.clip(round(13.0 * 86 / len(sub), 1), 7.5, 13.5))
        pnl = (chr(ord(panel) + i) if (panel and split > 1) else panel)
        part = (i + 1) if split > 1 else None
        out = _outname(tag, mode, labels, scale, part, order)
        extra_bits = ([f"part {i + 1}/{len(parts)} · {len(sub)} phenotypes"] if split > 1 else [])
        if order_note:
            extra_bits.append(order_note)
        extra = " · ".join(extra_bits)
        _draw_wheel(lay, arms, span, scale, pnl, out, fs, note_extra=extra)
        lay.to_csv(str(out).replace(".png", "_layout.csv"), index=False)
        parts_data.append((lay, span, fs, pnl or chr(ord("a") + i)))
        print(f"wrote {out.name}  ({len(sub)} spokes, {sub['category'].nunique()} sectors, "
              f"spoke_fs={fs}, cats={cats})")

    if grid:
        if len(parts_data) != 4:
            print(f"[grid] note: {len(parts_data)} parts (expected 4 for a 2×2) — tiling anyway")
        gout = Path(str(_outname(tag, mode, labels, scale, None, order)).replace(".png", "_2x2.png"))
        # Scale and ordering are disclosed in the manuscript caption; keep the grid itself clean.
        combine_grid(parts_data, arms, scale, gout)
        print(f"wrote {gout.name}  (2×2 grid of {len(parts_data[:4])} wheels)")

    macro = "  ".join(f"{k}={df[f'auc_{k}'].mean():.3f}" for k in keys)
    print(f"[{labels} | mode={mode} | scale={scale} | split={split} | order={order} | "
          f"lp_source={resolved_lp_source}] macro {macro} "
          f"over {len(df)} phenotypes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctclip", choices=list(MODE_SPEC))
    ap.add_argument("--scale", default="per_spoke", choices=["per_spoke", "shared"])
    ap.add_argument("--labels", default="86", choices=["86", "all221"])
    ap.add_argument("--split", type=int, default=1, help="fan the phenotypes across N category-balanced wheels")
    ap.add_argument("--tag", default="f2llm")
    ap.add_argument("--panel", default="", help="bold panel letter; with --split, parts get consecutive letters")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--grid", action="store_true",
                    help="also tile the (≤4) split wheels into one 2×2 figure with a shared legend")
    ap.add_argument("--order", default="ctpa", choices=["ctpa", "advantage"],
                    help="panel/group priority: original CTPA relevance or f2llm AUROC advantage")
    ap.add_argument("--lp-source", default="auto", choices=["auto", "seed42", "perseed20"],
                    help="all221 linear-probe source; auto preserves legacy order-dependent behaviour")
    plot(**vars(ap.parse_args()))
