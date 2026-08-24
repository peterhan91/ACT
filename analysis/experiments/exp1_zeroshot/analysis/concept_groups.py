"""
Anatomical category maps for the circular maps — one per label set.

The circular plot lays concepts out as wedges grouped into sectors; the sector =
an anatomical category. `GROUPS[<set>]` is an ordered {category: [labels...]} map
covering EXACTLY the labels scored for that set (chest18 / abd30 / rsna9).

`DISPLAY` gives a prettier label for the wedge tick (defaults to the raw label).
`assign_group(label, set_key)` returns the category; `group_order(set_key)` gives
the sector order (used consistently across datasets that share a label set).

Edit these maps freely — they are presentation only and do not touch the scores.
"""
from __future__ import annotations

# ── chest 18 (CT-RATE / RAD-ChestCT / PMBB-chest share LABELS_18) ─────────────
CHEST18 = {
    "Cardiac":       ["Cardiomegaly", "Pericardial effusion",
                      "Coronary artery wall calcification"],
    "Vascular":      ["Arterial wall calcification"],
    "Mediastinal":   ["Lymphadenopathy"],
    "Airway":        ["Peribronchial thickening", "Bronchiectasis"],
    "Parenchymal":   ["Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
                      "Pulmonary fibrotic sequela", "Mosaic attenuation pattern",
                      "Consolidation", "Interlobular septal thickening"],
    "Pleural":       ["Pleural effusion"],
    "Miscellaneous": ["Medical material", "Hiatal hernia"],
}

# ── abdomen 30 (PMBB-abd, Merlin-style findings) ─────────────────────────────
ABD30 = {
    "Hepatobiliary": ["Biliary Ductal Dilation", "Hepatomegaly", "Hepatic Steatosis",
                      "Surgically Absent Gallbladder", "Gallstones"],
    "Pancreas/Spleen": ["Pancreatic Atrophy", "Splenomegaly"],
    "Genitourinary": ["Renal Cyst", "Renal Hypodensity", "Hydronephrosis",
                      "Prostatomegaly"],
    "Gastrointestinal": ["Hiatal Hernia", "Submucosal Edema", "Bowel Obstruction",
                         "Appendicitis", "Free Air"],
    "Peritoneal":    ["Ascites", "Anasarca"],
    "Vascular":      ["Atherosclerosis", "Thrombosis", "Abdominal Aortic Aneurysm",
                      "Coronary Calcification", "Aortic Valve Calcification"],
    "Osseous":       ["Fracture", "Osteopenia"],
    "Lymph/Neoplasm": ["Lymphadenopathy", "Metastatic Disease"],
    "Thoracic":      ["Atelectasis", "Pleural Effusion", "Cardiomegaly"],
}

# ── RSNA-2023 abdominal trauma (9) ───────────────────────────────────────────
RSNA9 = {
    "Bowel/Mesentery": ["bowel_injury", "extravasation_injury"],
    "Kidney":          ["kidney_low", "kidney_high"],
    "Liver":           ["liver_low", "liver_high"],
    "Spleen":          ["spleen_low", "spleen_high"],
    "Overall":         ["any_injury"],
}

GROUPS = {"chest18": CHEST18, "abd30": ABD30, "rsna9": RSNA9}

# Pretty wedge labels (raw label -> display); unlisted labels are Title-cased as-is.
# RSNA-2023 names mirror the exact zero-shot prompts (clear3d/paths.py): bowel &
# extravasation are binary; kidney/liver/spleen have low/high injury-severity grades.
DISPLAY = {
    "bowel_injury": "Bowel injury", "extravasation_injury": "Active extravasation",
    "kidney_low": "Low-grade kidney injury", "kidney_high": "High-grade kidney injury",
    "liver_low": "Low-grade liver injury", "liver_high": "High-grade liver injury",
    "spleen_low": "Low-grade spleen injury", "spleen_high": "High-grade spleen injury",
    "any_injury": "Any abdominal injury",
}


# Short sector labels for the crowded inner ring (full name -> short). Anything
# not listed falls back to the full category name.
CATEGORY_SHORT = {
    # chest
    "Mediastinal": "Medias.", "Parenchymal": "Parench.", "Miscellaneous": "Misc.",
    # abdomen
    "Hepatobiliary": "Hepato.", "Pancreas/Spleen": "Panc/Spl",
    "Genitourinary": "GU", "Gastrointestinal": "GI", "Peritoneal": "Perit.",
    "Lymph/Neoplasm": "Lymph", "Osseous": "Bone",
    # rsna
    "Bowel/Mesentery": "Bowel",
}


def short_category(name: str) -> str:
    return CATEGORY_SHORT.get(name, name)


# Global category -> arc color, so every category has ONE fixed color across all panels
# (e.g. Vascular is always orange). Chest keeps the approved Set1 palette; abdomen/RSNA
# get vivid, distinct colors (the abdomen+RSNA composite needs all 14 to be distinct).
CATEGORY_COLORS = {
    # chest (approved Set1 palette)
    "Miscellaneous":    "#E41A1C",  # red
    "Mediastinal":      "#377EB8",  # blue
    "Parenchymal":      "#4DAF4A",  # green
    "Airway":           "#984EA3",  # purple
    "Vascular":         "#FF7F00",  # orange  (SHARED chest + abdomen)
    "Cardiac":          "#FFFF33",  # yellow
    "Pleural":          "#A65628",  # brown
    # abdomen
    "Hepatobiliary":    "#E41A1C",  # red
    "Pancreas/Spleen":  "#FFD92F",  # gold
    "Genitourinary":    "#377EB8",  # blue
    "Gastrointestinal": "#4DAF4A",  # green
    "Peritoneal":       "#984EA3",  # purple
    "Osseous":          "#A65628",  # brown
    "Lymph/Neoplasm":   "#E7298A",  # magenta
    "Thoracic":         "#1B9E77",  # teal
    # rsna (distinct from abdomen so the combined circle's 14 sectors stay distinct)
    "Kidney":           "#17BECF",  # cyan
    "Liver":            "#F781BF",  # pink
    "Spleen":           "#BCBD22",  # olive
    "Bowel/Mesentery":  "#6A3D9A",  # deep purple
    "Overall":          "#B15928",  # rust
}


def category_color(name: str) -> str:
    return CATEGORY_COLORS.get(name, "#999999")   # gray fallback for any unmapped category


def group_order(set_key: str) -> list[str]:
    return list(GROUPS[set_key].keys())


def assign_group(label: str, set_key: str) -> str:
    for group, labels in GROUPS[set_key].items():
        if label in labels:
            return group
    return "Other"


def display_label(label: str) -> str:
    return DISPLAY.get(label, label[:1].upper() + label[1:])
