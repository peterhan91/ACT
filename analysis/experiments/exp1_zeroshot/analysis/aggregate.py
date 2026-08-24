#!/usr/bin/env python3
"""
Aggregate exp1 zero-shot results into clean, plot-ready tables.

Reads, for every (dataset, model) pair, the per-model result JSON in its CANONICAL
mode (config.MODELS[m]["mode"]: ours=plain, baselines=native) and emits under
analysis/tables/:

  per_label_auc__<dataset>.csv   one row / label: label, group, auc_<model>...
                                 (missing model -> empty cell)
  mean_auc_matrix.csv            datasets x models, mean test AUROC
  model_ranking.csv              models ranked by CT-RATE mean AUROC (plot order)

Run:  python analysis/aggregate.py
"""
from __future__ import annotations

import json
import sys
import pandas as pd

import config
import concept_groups as cg


def load_per_label(dataset: str, model_dir: str, mode: str) -> dict | None:
    """{label: auc} for one model on one dataset, or None if the JSON is absent."""
    path = config.result_json(model_dir, dataset, mode)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return dict(zip(d["labels"], [d["per_label_auc"][l] for l in d["labels"]])) \
        if isinstance(d["per_label_auc"], dict) else dict(zip(d["labels"], d["per_label_auc"]))


def build_dataset_table(dataset: str, spec: dict) -> pd.DataFrame:
    """Per-label AUROC table (label, group, auc_<model>...) for one dataset."""
    set_key = spec["groups"]
    # label order = the order they appear in the category map (sector layout)
    ordered = [l for g in cg.group_order(set_key) for l in cg.GROUPS[set_key][g]]

    rows = {l: {"label": l, "group": cg.assign_group(l, set_key),
                "display": cg.display_label(l)} for l in ordered}
    for model_dir, meta in config.MODELS.items():
        per_label = load_per_label(dataset, model_dir, meta["mode"])
        if per_label is None:
            continue
        for l, auc in per_label.items():
            rows.setdefault(l, {"label": l,
                                "group": cg.assign_group(l, set_key),
                                "display": cg.display_label(l)})
            rows[l][f"auc_{model_dir}"] = auc
    df = pd.DataFrame([rows[l] for l in ordered if l in rows])
    missing = [l for l in rows if l not in ordered]
    if missing:
        print(f"  [{dataset}] WARNING: {len(missing)} scored labels not in "
              f"concept_groups[{set_key}]: {missing}", file=sys.stderr)
        df = pd.concat([df, pd.DataFrame([rows[l] for l in missing])], ignore_index=True)
    return df


def _reference_dataset(set_key: str) -> str:
    """Dataset whose ours-AUROC fixes the within-sector label order for a label set.
    Uses RANK_DATASET (CT-RATE) when it owns the set, else the sole dataset using it."""
    if config.DATASETS[config.RANK_DATASET]["groups"] == set_key:
        return config.RANK_DATASET
    return next(d for d, s in config.DATASETS.items() if s["groups"] == set_key)


def _label_order(ref_df, set_key: str) -> list:
    """Full wedge order: categories by ASCENDING mean ours AUROC, then labels within
    each category by ASCENDING ours AUROC (so the larger AUROC sits clockwise)."""
    groups = [g for g in cg.group_order(set_key) if (ref_df["group"] == g).any()]
    if "auc_ours" in ref_df.columns:
        cat_auc = {g: ref_df.loc[ref_df["group"] == g, "auc_ours"].mean() for g in groups}
        groups.sort(key=lambda g: cat_auc[g])           # categories ascending
    order = []
    for g in groups:
        sub = ref_df[ref_df["group"] == g]
        if "auc_ours" in sub.columns:
            sub = sub.sort_values("auc_ours", ascending=True, kind="stable")  # labels ascending
        order.extend(sub["label"].tolist())
    order += [l for l in ref_df["label"] if l not in order]   # any uncovered -> keep last
    return order


def main() -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dfs = {d: build_dataset_table(d, s) for d, s in config.DATASETS.items()}

    # Canonical within-sector label order per label set: by ours AUROC on the
    # set's reference dataset, reused across every dataset sharing the set so a finding
    # keeps the same wedge angle across plots.
    set_order = {}
    for set_key in dict.fromkeys(s["groups"] for s in config.DATASETS.values()):
        ref = _reference_dataset(set_key)
        order = _label_order(dfs[ref], set_key)
        set_order[set_key] = order
        grp = dict(zip(dfs[ref]["label"], dfs[ref]["group"]))
        pd.DataFrame({"order": range(1, len(order) + 1), "label": order,
                      "group": [grp[l] for l in order], "ref_dataset": ref}) \
          .to_csv(config.TABLES_DIR / f"label_order__{set_key}.csv", index=False)

    mean_rows = []
    ordered = {}   # reordered per-dataset frames, reused to assemble composites
    for dataset, spec in config.DATASETS.items():
        rank_key = {l: i for i, l in enumerate(set_order[spec["groups"]])}
        df = (dfs[dataset].assign(_k=lambda d: d["label"].map(rank_key))
              .sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True))
        ordered[dataset] = df
        out = config.TABLES_DIR / f"per_label_auc__{dataset}.csv"
        df.to_csv(out, index=False)
        present = [m for m in config.MODELS if f"auc_{m}" in df.columns]
        print(f"{dataset:14s} {len(df):3d} labels x {len(present)} models -> {out.name}")
        for m in config.MODELS:
            col = f"auc_{m}"
            mean_rows.append(dict(dataset=dataset, model=m,
                                  mean_auc=df[col].mean() if col in df.columns else float("nan")))

    # Composite tables: concatenate member datasets' (already-ordered) rows into one
    # circle, tagging each concept with its source dataset (two contiguous blocks).
    for cname, cspec in getattr(config, "COMPOSITES", {}).items():
        parts = [ordered[m].assign(dataset=m) for m in cspec["members"]]
        comb = pd.concat(parts, ignore_index=True)
        comb.to_csv(config.TABLES_DIR / f"per_label_auc__{cname}.csv", index=False)
        print(f"{cname:14s} {len(comb):3d} concepts ({' + '.join(cspec['members'])})")

    mean = pd.DataFrame(mean_rows)
    matrix = mean.pivot(index="dataset", columns="model", values="mean_auc")
    matrix = matrix.reindex(index=list(config.DATASETS), columns=list(config.MODELS))
    matrix.to_csv(config.TABLES_DIR / "mean_auc_matrix.csv")

    rank = (mean[mean.dataset == config.RANK_DATASET][["model", "mean_auc"]]
            .dropna().sort_values("mean_auc", ascending=False).reset_index(drop=True))
    rank["rank"] = rank.index + 1
    rank["display"] = rank["model"].map(lambda m: config.MODELS[m]["display"])
    rank = rank[["rank", "model", "display", "mean_auc"]]
    rank.to_csv(config.TABLES_DIR / "model_ranking.csv", index=False)

    print(f"\nCT-RATE ranking (plot order, outer -> inner):")
    print(rank.to_string(index=False))
    print(f"\nWrote tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
