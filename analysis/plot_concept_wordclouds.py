"""Word-cloud atlas of the 376,194 unique radiological concepts.

Reproduces the look of Extended Data Fig. 1 (Lu et al., Nat Med 2024): one
word cloud per organ-system category plus an "All categories" panel, larger
words = more frequent, laid out on a grid with the category name and concept
count beneath each panel.

Concepts come from concept_bank.f2llm_emb.npz (the default f2llm bank); the
organ-system assignment is produced by concept_categories.py.

Usage:
    python plot_concept_wordclouds.py
Output:
    figures/concept_wordclouds.png  (+ .pdf)
"""
import os
import collections
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud, STOPWORDS

from concept_categories import categorize, CATEGORIES

BANK = "concept_bank.f2llm_emb.npz"
OUT = "figures/concept_wordclouds"
SEED = 7
MAX_TEXT_PER_CAT = 70_000   # subsample large categories for speed (freqs preserved)

# ---------------------------------------------------------------- stopwords
_UNITS_POS = """
mm cm ml cc x ap pa si measuring measures measure measured diameter
approximately approx size sized number numerous several multiple
right left bilateral bilaterally upper lower mid middle anterior posterior
superior inferior proximal distal medial lateral central peripheral
region regions area areas segment segments segmental level levels zone zones
aspect portion portions side sides
""".split()

_FILLER = """
likely appears appear appearing seen noted note redemonstrated redemonstration
demonstrated demonstrates demonstrate demonstrating evidence finding findings
status post present unchanged stable prior priors compared comparison
new old slightly mildly relatively overall including without within associated
consistent suggestive representing represents related versus otherwise grossly
again similar additional additionally adjacent also well known unremarkable
normal abnormal no not nonspecific non specific concerning compatible favored
favor probable possible possibly may could would suspected suspicious
better worse interval change changed mild moderate minimal severe marked
small large tiny minimally quadrant structure structures appearance amount
containing involving study observed evaluation evaluate evaluated partially
visualized imaged characterize characterized dimension dimensions contour
contours caliber redemo gas containing trace
""".split()

STOP = set(STOPWORDS) | set(_UNITS_POS) | set(_FILLER)

# ---------------------------------------------------------------- palette (EDF1-like)
# saturated accents (used for the big words) + lighter pinks (the bulk)
_PALETTE = [
    ("#B5179E", 3),   # magenta
    ("#7B2D8E", 3),   # purple
    ("#1CA4DE", 3),   # cyan / blue
    ("#D6336C", 2),   # deep pink
    ("#E27DAE", 5),   # mid pink
    ("#EFA9C4", 4),   # light pink
    ("#9B5DE5", 2),   # violet
    ("#4DA3D9", 2),   # steel blue
]
_COLORS = [c for c, w in _PALETTE for _ in range(w)]


def color_func(word, font_size, position, orientation, font_path, random_state=None,
               **kw):
    rs = random_state if random_state is not None else np.random
    # bias the largest words toward the saturated accents
    if font_size > 70:
        return rs.choice(["#B5179E", "#7B2D8E", "#1CA4DE", "#D6336C"])
    return rs.choice(_COLORS)


def load():
    z = np.load(BANK, allow_pickle=True)
    concepts = np.asarray([str(s) for s in z["concepts"]], dtype=object)
    cat_path = "outputs/_concept_category.npy"
    if os.path.exists(cat_path):
        cats = np.load(cat_path, allow_pickle=True)
        if len(cats) != len(concepts):
            cats = np.array([categorize(s) for s in concepts], dtype=object)
    else:
        cats = np.array([categorize(s) for s in concepts], dtype=object)
    return concepts, cats


def text_for(concepts, mask, rng):
    idx = np.where(mask)[0]
    if len(idx) > MAX_TEXT_PER_CAT:
        idx = rng.choice(idx, MAX_TEXT_PER_CAT, replace=False)
    return " ".join(concepts[i] for i in idx)


def make_cloud(text, max_words, w=1000, h=1000):
    wc = WordCloud(
        width=w, height=h, background_color="white",
        stopwords=STOP, prefer_horizontal=0.85, max_words=max_words,
        collocations=True, collocation_threshold=18, normalize_plurals=True,
        regexp=r"\b[a-zA-Z][a-zA-Z'\-]+\b", min_word_length=3,
        relative_scaling=0.55, min_font_size=8, random_state=SEED,
        margin=2, max_font_size=None,
    )
    wc.generate(text)
    wc.recolor(color_func=color_func, random_state=SEED)
    return wc.to_array()


def main():
    os.makedirs("figures", exist_ok=True)
    concepts, cats = load()
    rng = np.random.default_rng(SEED)

    counts = collections.Counter(cats)
    # 16 largest organ-system categories -> clean 4x4 grid (no "All" panel)
    order = [name for name, _ in CATEGORIES if counts.get(name, 0) >= 800]
    order.sort(key=lambda n: counts[n], reverse=True)
    order = order[:16]

    panels = [(name, cats == name, counts[name]) for name in order]

    n = len(panels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))

    print(f"rendering {n} panels ({nrow}x{ncol}) ...")
    clouds = []
    for i, (name, mask, cnt) in enumerate(panels):
        txt = text_for(concepts, mask, rng)
        clouds.append(make_cloud(txt, 130))
        print(f"  [{i+1}/{n}] {name:28s} {cnt:>8,}")

    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.5 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for i, (name, mask, cnt) in enumerate(panels):
        ax = axes.ravel()[i]
        ax.imshow(clouds[i], interpolation="bilinear")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#222"); sp.set_linewidth(0.8)
        ax.set_title(
            f"{name}\n({cnt:,})",
            fontsize=16, fontweight="normal",
            pad=6, linespacing=1.4,
        )

    for j in range(n, nrow * ncol):
        axes.ravel()[j].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                        wspace=0.06, hspace=0.34)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    main()
