"""Build trusted concept spaces for the 9 RSNA-2023 abdominal-trauma labels.

Mirrors the INSPECT phenotype track (`trusted_concept_space.py`) but with hand-
curated trauma profiles for the 9 RSNA-2023 binary labels. Reuses the shared
`GLOBAL_EXCLUDE`, regex/scan helpers, summary schema, and audit cross-check
machinery from the phenotype module.

Design (set by the user before this module was written):

  * shared base + grade modifiers
    Both `*_low` and `*_high` labels for kidney/liver/spleen include the same
    base organ-injury concepts; `direct` is filtered by low- vs high-grade
    modifier regex, ambiguous concepts go to `associated`.

  * `any_injury` = union of all 8 direct tiers + global catch-alls
    (hemoperitoneum, pneumoperitoneum, retroperitoneal hematoma, free
    intraperitoneal fluid, active extravasation).

  * The chest-CT-derived bank has many confounds for trauma terms (subcapsular
    hepatic cysts, perinephric stranding from ureterolithiasis, chronic
    splenic infarcts), so per-label `exclude` patterns suppress them.

Writes per-label trusted_concepts.csv + current_top10_audit.csv (against the
existing v1 / linear_openai 20-seed audit JSON) + summary.csv + manifest.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trusted_concept_space import (
    GLOBAL_EXCLUDE,
    LabelProfile,
    classify_concept,
    compile_all,
    load_audit,
    load_concepts,
    load_top10,
    profile_status,
    radiology_flag,
    scan_profile,
    slug,
    write_csv,
)


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Building blocks — shared base patterns and grade modifiers.
# ---------------------------------------------------------------------------

LOW_GRADE_MOD = (
    r"\b(small|tiny|minor|superficial|trace|minimal|early|focal|"
    r"grade i\b|grade 1\b|grade ii\b|grade 2\b)\b"
)
HIGH_GRADE_MOD = (
    r"\b(deep|extensive|large|complex|severe|shattered|fragmented|"
    r"devitalized|devascular\w*|multiple|comminuted|complete|"
    r"grade iii\b|grade 3\b|grade iv\b|grade 4\b|grade v\b|grade 5\b)\b"
)


def _organ_injury_base(*organ_alts: str) -> tuple[str, ...]:
    """Base patterns for an organ injury: laceration/hematoma/contusion in either
    word-order, plus subcapsular-hematoma and peri-organ bleeding tropes.
    """
    org = "(" + "|".join(organ_alts) + ")"
    return (
        rf"\b{org}\b.{{0,40}}\blacerat\w*\b",
        rf"\blacerat\w*\b.{{0,40}}\b{org}\b",
        rf"\b{org}\b.{{0,40}}\b(hematoma|hemorrhag\w*|contusion|injur\w*|fractur\w*|disrupt\w*)\b",
        rf"\b(hematoma|hemorrhag\w*|contusion|injur\w*)\b.{{0,40}}\b{org}\b",
    )


def _peri_organ_bleed(peri_alt: str) -> str:
    return rf"\b{peri_alt}\w*\b.{{0,40}}\b(hematoma|hemorrhag\w*|fluid|blood)\b"


def _with_low(base: tuple[str, ...]) -> tuple[str, ...]:
    """Require a low-grade modifier within 60 chars of each base pattern.

    Bounded-distance form (no regex lookahead) is ~50× faster on 376k concepts
    than `(?=.*MOD).*pat` and avoids catastrophic backtracking.
    """
    out: list[str] = []
    for pat in base:
        out.append(rf"{LOW_GRADE_MOD}.{{0,60}}{pat}")
        out.append(rf"{pat}.{{0,60}}{LOW_GRADE_MOD}")
    return tuple(out)


def _with_high(base: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for pat in base:
        out.append(rf"{HIGH_GRADE_MOD}.{{0,60}}{pat}")
        out.append(rf"{pat}.{{0,60}}{HIGH_GRADE_MOD}")
    return tuple(out)


# Active-extravasation / vascular-bleed patterns near an organ.
def _vascular_bleed(*organ_alts: str) -> tuple[str, ...]:
    org = "(" + "|".join(organ_alts) + ")"
    return (
        rf"\b(active (contrast )?extravasation|active (arterial )?bleed|blush of contrast|pseudoaneurysm)\b.{{0,60}}\b{org}\b",
        rf"\b{org}\b.{{0,60}}\b(active (contrast )?extravasation|active (arterial )?bleed|blush of contrast|pseudoaneurysm)\b",
    )


# Subcapsular hematoma scoped to an organ.
def _subcapsular_hematoma(*organ_alts: str) -> tuple[str, ...]:
    org = "(" + "|".join(organ_alts) + ")"
    return (
        rf"\bsubcapsular\b.{{0,30}}\b{org}\b.{{0,30}}\b(hematoma|hemorrhag\w*|fluid)\b",
        rf"\bsubcapsular\b.{{0,30}}\b(hematoma|hemorrhag\w*)\b.{{0,30}}\b{org}\b",
        rf"\b{org}\b.{{0,30}}\bsubcapsular\b.{{0,30}}\b(hematoma|hemorrhag\w*|fluid)\b",
    )


# ---------------------------------------------------------------------------
# Per-organ assembled patterns.
# ---------------------------------------------------------------------------

BASE_SPLEEN = _organ_injury_base("splen\\w*", "spleen")
BASE_LIVER = _organ_injury_base("liver", "hepat\\w*")
BASE_KIDNEY = _organ_injury_base("kidney", "kidneys", "renal")

PERISPLENIC_BLEED = _peri_organ_bleed("perisplen")
PERIHEPATIC_BLEED = _peri_organ_bleed("perihepat")
PERIRENAL_BLEED = (
    _peri_organ_bleed("perinephric"),
    _peri_organ_bleed("perirenal"),
)
PERIRENAL_BLEED = tuple(p for p in PERIRENAL_BLEED)

SUBCAP_SPLEEN = _subcapsular_hematoma("splen\\w*", "spleen")
SUBCAP_LIVER = _subcapsular_hematoma("liver", "hepat\\w*")
SUBCAP_KIDNEY = _subcapsular_hematoma("kidney", "kidneys", "renal")

VASC_SPLEEN = _vascular_bleed("splen\\w*", "spleen")
VASC_LIVER = _vascular_bleed("liver", "hepat\\w*")
VASC_KIDNEY = _vascular_bleed("kidney", "kidneys", "renal")

# Bowel: perforation, mesenteric injury, bowel-wall hematoma/contusion.
# Mesenteric stranding alone is non-specific (ischemia, pancreatitis, ileus,
# peritonitis) — keep it in BASE_BOWEL_ASSOC, NOT direct. Direct requires
# explicit injury/perforation/hematoma/extravasation wording.
BASE_BOWEL = (
    r"\bbowel\b.{0,30}\b(perforat\w*|injur\w*|disrupt\w*|discontin\w*|hematoma|contusion)\b",
    r"\b(perforat\w*|injur\w*|disrupt\w*|hematoma|contusion)\b.{0,30}\bbowel\b",
    r"\b(mesenter\w*)\b.{0,40}\b(hematoma|injur\w*|extravasation|hemorrhag\w*|disrupt\w*)\b",
    r"\b(small bowel|large bowel|jejun\w*|ileum|duoden\w*|colon\w*)\b.{0,40}\b(perforat\w*|injur\w*|hematoma|disrupt\w*|extravasation)\b",
    r"\b(active (contrast )?extravasation|blush of contrast)\b.{0,60}\b(bowel|mesenter\w*|small bowel|jejun\w*|ileum)\b",
)
# Associated for bowel — non-specific signs that can flag bowel injury but
# also have many non-trauma causes.
BASE_BOWEL_ASSOC = (
    r"\bmesenteric stranding\b",
    r"\bmesentery stranding\b",
    r"\bmesenteric edema\b",
    r"\bbowel wall thickening\b",
    r"\bfree (intraperitoneal|intra[- ]abdominal) fluid\b.{0,40}\bpelvis\b",
    r"\bpneumoperitoneum\b",
    r"\bfree air\b.{0,30}\b(abdomen|peritone\w*)\b",
)

# Generic free-air/free-fluid markers used as bowel-injury associates and as
# any_injury catch-alls.
FREE_AIR_FLUID = (
    r"\bpneumoperitoneum\b",
    r"\bfree (intraperitoneal|intra[- ]abdominal) air\b",
    r"\bfree air\b.{0,30}\b(abdomen|peritone\w*)\b",
    r"\bhemoperitoneum\b",
    r"\bfree (intraperitoneal|intra[- ]abdominal) fluid\b",
    r"\bretroperitoneal hematoma\b",
)

# Extravasation as its own primary finding (organ-agnostic).
EXTRAV_DIRECT = (
    r"\bactive (contrast )?extravasation\b",
    r"\bactive (arterial )?bleed\b",
    r"\bcontrast extravasation\b",
    r"\bblush of contrast\b",
    r"\bjet of contrast\b",
    r"\bpseudoaneurysm\b",
)
EXTRAV_ASSOC = (
    r"\b(contained|expanding|enlarging)\b.{0,30}\bhematoma\b",
    r"\bcoils?\b.{0,20}\bembolization\b",  # often co-described with prior bleed
    r"\bretroperitoneal hematoma\b",
    r"\bhemoperitoneum\b",
)
EXTRAV_EXCLUDE = (
    r"\bcontrast (within|in) the bowel lumen\b",
    r"\boral contrast\b",
    r"\bintravesical contrast\b",
    r"\bintravascular contrast\b",
    r"\bcontrast in the renal collecting system\b",
    r"\bchronic\b",
    r"\bthrombosed\b",
    r"\bcalcified\b",
    r"\bcoil\w*\b.{0,30}\bemboliz\w*\b",
    r"\bemboliz\w*\b.{0,30}\bcoil\w*\b",
    r"\btreated\b",
    r"\bpost[- ]?treatment\b",
    r"\bsurgical clip\w*\b",
    r"\bold\b",
    r"\bremote\b",
    r"\bpost[- ]?surgical\b",
    r"\bneobladder\b",
    r"\bileal conduit\b",
    r"\bgraft\b",
    r"\banastomos\w*\b",
    r"\b(bowel|colon|jejun|ileum) lumen\b",
    r"\bcontrast (within|in|into) (the )?(bowel|colon|jejun|ileum|stomach|duoden)\w*\b",
    r"\blimited evaluation\b",
)


# Organ-specific excludes — suppress chest-CT confounds (cysts, hemangiomas,
# benign focal lesions, post-surgical anatomy, chronic infarcts, stones).
SPLEEN_EXCLUDE = (
    r"\bsplenic cyst\w*\b",
    r"\bsplenomegaly\b",
    r"\bsplenectomy\b",
    r"\bpost[- ]?splenectomy\b",
    r"\bhemangioma\b",
    r"\blymphoma\b",
    r"\bmetasta\w*\b",
    r"\baccessory spleen\b",
    r"\bsplenule\b",
    r"\bcalcified granuloma\b",
    r"\bgranulomata?\b",
    r"\bchronic\b",
    r"\binfarct\w*\b",  # 330 hits, mostly chronic/embolic — exclude from direct/assoc here
    r"\bbenign\b",
    r"\bindeterminate (lesion|hypodensity)\b",
    r"\blikely (hemangioma|cyst|benign)\b",
)
LIVER_EXCLUDE = (
    r"\bhepatic cyst\w*\b",
    r"\bliver cyst\w*\b",
    r"\babscess\b",
    r"\bbiloma\b",
    r"\bgas in (the )?(collection|fluid|hematoma)\b",
    r"\b(locules?|foci) of gas\b",
    r"\brim[- ]?enhanc\w*\b.{0,30}\bgas\b",
    r"\bold injur\w*\b",
    r"\bremote injur\w*\b",
    r"\bhemangioma\b",
    r"\bfnh\b",
    r"\bfocal nodular hyperplasia\b",
    r"\bsteatosis\b",
    r"\bfatty (liver|infiltration)\b",
    r"\bcirrhosis\b",
    r"\bcirrhotic\b",
    r"\bmetasta\w*\b",
    r"\bhcc\b",
    r"\bhepatocellular carcinoma\b",
    r"\bbiliary\b",
    r"\bbile duct\b",
    r"\bgallbladder\b",
    r"\bchronic\b",
    r"\bbenign\b",
    r"\bindeterminate (lesion|hypodensity)\b",
    r"\blikely (hemangioma|cyst|benign|fnh)\b",
)
KIDNEY_EXCLUDE = (
    r"\bcyst\w*\b",                     # blanket exclude — trauma rarely co-described with cysts
    r"\bpolycystic\b",
    r"\bpkd\b",
    r"\bcystic disease\b",
    r"\bold injur\w*\b",
    r"\bremote injur\w*\b",
    r"\bbosniak\b",
    r"\bhydronephrosis\b",
    r"\bureterolithiasis\b",
    r"\bnephrolithiasis\b",
    r"\brenal calcul\w*\b",
    r"\bureter\w* (stone|calcul\w*)\b",
    r"\b(stone|calcul\w*) in (the )?(ureter|renal pelvis|kidney)\b",
    r"\bobstructive (calcul\w*|stone|uropathy)\b",
    r"\bperinephric stranding\b",
    r"\bperiureter\w*\b",
    r"\bhydroureter\w*\b",
    r"\bmetasta\w*\b",
    r"\brcc\b",
    r"\brenal cell carcinoma\b",
    r"\btransplant\b",
    r"\bnephrectomy\b",
    r"\bchronic\b",
    r"\batrophic\b",
    r"\bbenign\b",
    r"\blikely (cyst|benign|bosniak)\b",
)
BOWEL_EXCLUDE = (
    r"\bileus\b",
    r"\bbowel obstruct\w*\b",
    r"\binflammatory bowel\b",
    r"\bcrohn\w*\b",
    r"\bulcerative colitis\b",
    r"\bdiverticulitis\b",
    r"\bappendicitis\b",
    r"\banastomot\w*\b",
    r"\bcolostomy\b",
    r"\bileostomy\b",
    r"\bchronic\b",
)


# ---------------------------------------------------------------------------
# Build the 9 LabelProfile instances.
# ---------------------------------------------------------------------------

def _profile(label: str, *, direct, associated, exclude, notes: str) -> LabelProfile:
    return LabelProfile(
        label=label,
        direct=tuple(direct),
        associated=tuple(associated),
        exclude=tuple(exclude),
        notes=notes,
        source="rsna2023_curated",
    )


PROF_SPLEEN_LOW = _profile(
    "spleen_low",
    direct=(_with_low(BASE_SPLEEN) + SUBCAP_SPLEEN + (
        r"\b(splen\w*|spleen)\b.{0,30}\bcontusion\b",
        r"\bcontusion\b.{0,30}\b(splen\w*|spleen)\b",
    )),
    associated=(BASE_SPLEEN + (PERISPLENIC_BLEED,)),
    exclude=SPLEEN_EXCLUDE + VASC_SPLEEN + _with_high(BASE_SPLEEN) + _with_high(SUBCAP_SPLEEN),
    notes="Low-grade splenic injury: subcapsular hematoma, small/minor/grade I-II laceration or contusion. High-grade modifiers (deep/extensive/large/grade III-V) and vascular bleed pushed to exclude.",
)
PROF_SPLEEN_HIGH = _profile(
    "spleen_high",
    direct=(_with_high(BASE_SPLEEN) + VASC_SPLEEN + (
        r"\bshattered (spleen|splen\w*)\b",
        r"\b(splen\w*|spleen)\b.{0,30}\b(devascular\w*|devitalized|fragmented|comminuted)\b",
        r"\b(devascular\w*|devitalized|fragmented).{0,30}\b(splen\w*|spleen)\b",
        r"\b(splen\w*|spleen)\b.{0,40}\b(vascular pedicle|hilar|hilum)\b.{0,40}\b(injur\w*|disrupt\w*)\b",
    )),
    associated=(BASE_SPLEEN + (PERISPLENIC_BLEED,)),
    exclude=SPLEEN_EXCLUDE + _with_low(BASE_SPLEEN),
    notes="High-grade splenic injury: deep/extensive/grade III-V laceration, shattered spleen, vascular pedicle injury, active extravasation or pseudoaneurysm.",
)

PROF_LIVER_LOW = _profile(
    "liver_low",
    direct=(_with_low(BASE_LIVER) + SUBCAP_LIVER + (
        r"\b(liver|hepat\w*)\b.{0,30}\bcontusion\b",
        r"\bcontusion\b.{0,30}\b(liver|hepat\w*)\b",
    )),
    associated=(BASE_LIVER + (PERIHEPATIC_BLEED,)),
    exclude=LIVER_EXCLUDE + VASC_LIVER + _with_high(BASE_LIVER) + _with_high(SUBCAP_LIVER),
    notes="Low-grade hepatic injury: subcapsular hematoma, small/superficial/grade I-II laceration or contusion. High-grade modifiers (deep/extensive/large/grade III-V) on subcapsular collections and vascular bleed pushed to exclude.",
)
PROF_LIVER_HIGH = _profile(
    "liver_high",
    direct=(_with_high(BASE_LIVER) + VASC_LIVER + (
        r"\b(liver|hepat\w*)\b.{0,40}\b(vascular pedicle|hilar|hilum)\b.{0,40}\b(injur\w*|disrupt\w*)\b",
        r"\b(juxtahepatic|retrohepatic) (venous|ivc|inferior vena cava)\b.{0,40}\binjur\w*",
        r"\b(liver|hepat\w*)\b.{0,30}\b(devascular\w*|devitalized|fragmented|burst)\b",
        r"\b(devascular\w*|devitalized|fragmented).{0,30}\b(liver|hepat\w*)\b",
    )),
    associated=(BASE_LIVER + (PERIHEPATIC_BLEED,)),
    exclude=LIVER_EXCLUDE + _with_low(BASE_LIVER),
    notes="High-grade hepatic injury: deep/extensive/grade III-V laceration, juxtahepatic IVC/venous injury, active extravasation or pseudoaneurysm.",
)

PROF_KIDNEY_LOW = _profile(
    "kidney_low",
    direct=(_with_low(BASE_KIDNEY) + SUBCAP_KIDNEY + (
        r"\b(kidney|kidneys|renal)\b.{0,30}\bcontusion\b",
        r"\bcontusion\b.{0,30}\b(kidney|kidneys|renal)\b",
        r"\bsmall\b.{0,20}\bcortical defect\b.{0,30}\b(kidney|renal)\b",
    )),
    associated=(BASE_KIDNEY + PERIRENAL_BLEED),
    exclude=KIDNEY_EXCLUDE + VASC_KIDNEY + _with_high(BASE_KIDNEY) + _with_high(SUBCAP_KIDNEY) + (
        r"\burinoma\b",
        r"\brenal pedicle\b",
        r"\bshattered (kidney|renal)\b",
    ),
    notes="Low-grade renal injury: subcapsular hematoma, small/superficial/grade I-II laceration or contusion. Urinoma/pedicle/vascular bleed and high-grade modifiers on subcapsular collections pushed to exclude.",
)
PROF_KIDNEY_HIGH = _profile(
    "kidney_high",
    direct=(_with_high(BASE_KIDNEY) + VASC_KIDNEY + (
        r"\bshattered (kidney|renal)\b",
        r"\b(kidney|renal)\b.{0,30}\b(devascular\w*|devitalized|fragmented)\b",
        r"\b(devascular\w*|devitalized|fragmented).{0,30}\b(kidney|renal)\b",
        r"\b(renal pedicle|renal vascular pedicle)\b.{0,30}\b(injur\w*|disrupt\w*|laceration|avulsion)\b",
        r"\burinoma\b",
        r"\b(kidney|renal)\b.{0,40}\bcollecting system\b.{0,40}\b(injur\w*|disrupt\w*|extravasation)\b",
    )),
    associated=(BASE_KIDNEY + PERIRENAL_BLEED),
    exclude=KIDNEY_EXCLUDE + _with_low(BASE_KIDNEY),
    notes="High-grade renal injury: deep/extensive/grade III-V laceration, devascularized/shattered kidney, renal pedicle injury, urinoma, collecting-system disruption, active extravasation.",
)

PROF_BOWEL = _profile(
    "bowel_injury",
    direct=BASE_BOWEL + (
        r"\bbowel wall\b.{0,20}\b(hematoma|discontinuit\w*|disrupt\w*)\b",
        r"\b(small bowel|mesenter\w*)\b.{0,40}\b(active )?extravasation\b",
        r"\btransmural\b.{0,30}\b(injur\w*|disrupt\w*|laceration)\b.{0,30}\bbowel\b",
    ),
    associated=BASE_BOWEL_ASSOC,
    exclude=BOWEL_EXCLUDE,
    notes="Bowel/mesenteric injury direct: perforation, wall hematoma, mesenteric extravasation/hemorrhage. Associated: mesenteric stranding alone, bowel wall thickening, pneumoperitoneum, free intraperitoneal fluid (all non-specific without explicit injury wording). Excludes ileus, IBD, obstruction, anastomotic leak.",
)

PROF_EXTRAV = _profile(
    "extravasation_injury",
    direct=EXTRAV_DIRECT,
    associated=EXTRAV_ASSOC,
    exclude=EXTRAV_EXCLUDE,
    notes="Active arterial extravasation: contrast blush/jet, active bleed, pseudoaneurysm. Excludes contrast within bowel lumen / normal vascular phase.",
)

PROF_ANY = _profile(
    "any_injury",
    direct=(
        BASE_SPLEEN + BASE_LIVER + BASE_KIDNEY + BASE_BOWEL +
        EXTRAV_DIRECT + FREE_AIR_FLUID +
        (PERISPLENIC_BLEED, PERIHEPATIC_BLEED) + PERIRENAL_BLEED +
        SUBCAP_SPLEEN + SUBCAP_LIVER + SUBCAP_KIDNEY +
        VASC_SPLEEN + VASC_LIVER + VASC_KIDNEY
    ),
    associated=(
        r"\bfree fluid\b",
        r"\bcomplex fluid collection\b",
    ),
    exclude=tuple(set(
        SPLEEN_EXCLUDE + LIVER_EXCLUDE + KIDNEY_EXCLUDE + BOWEL_EXCLUDE + EXTRAV_EXCLUDE
    )),
    notes="Meta-label = union of 8 organ-injury direct tiers + global trauma catch-alls (hemoperitoneum, pneumoperitoneum, retroperitoneal hematoma, free intraperitoneal fluid, active extravasation).",
)


RSNA2023_PROFILES = {
    "bowel_injury": PROF_BOWEL,
    "extravasation_injury": PROF_EXTRAV,
    "kidney_low": PROF_KIDNEY_LOW,
    "kidney_high": PROF_KIDNEY_HIGH,
    "liver_low": PROF_LIVER_LOW,
    "liver_high": PROF_LIVER_HIGH,
    "spleen_low": PROF_SPLEEN_LOW,
    "spleen_high": PROF_SPLEEN_HIGH,
    "any_injury": PROF_ANY,
}

RSNA2023_LABELS = list(RSNA2023_PROFILES)


# ---------------------------------------------------------------------------
# Driver — mirrors trusted_concept_space.main() but for the 9 trauma labels.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_bank", default=str(ROOT / "concept_bank.openai_emb.npz"))
    ap.add_argument(
        "--audit_json",
        default=str(ROOT / "outputs/v1/audit/rsna2023__linear_openai__concept_importance.json"),
        help="Existing v1/linear_openai 20-seed audit; used for the current_top10 trust comparison.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(ROOT / "outputs/v1/audit/trusted_concept_spaces_rsna2023"),
    )
    ap.add_argument(
        "--labels", nargs="*", default=None,
        help="Subset of the 9 trauma labels to process; default is all.",
    )
    ap.add_argument(
        "--max_per_tier", type=int, default=0,
        help="Maximum rows per trusted tier; 0 keeps all matching concepts.",
    )
    args = ap.parse_args()

    concept_path = Path(args.concept_bank)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    concepts = load_concepts(concept_path)
    audit = load_audit(Path(args.audit_json)) if Path(args.audit_json).exists() else {}
    labels = args.labels if args.labels is not None else RSNA2023_LABELS
    unknown = [l for l in labels if l not in RSNA2023_PROFILES]
    if unknown:
        raise SystemExit(f"Unknown RSNA-2023 labels: {unknown}; choose from {RSNA2023_LABELS}")

    top10_by_label = load_top10(audit)
    global_exclude = compile_all(GLOBAL_EXCLUDE)

    summary_rows: list[dict] = []
    manifest: dict[str, object] = {
        "concept_bank": str(concept_path),
        "audit_json": str(Path(args.audit_json)),
        "n_concepts": len(concepts),
        "n_labels": len(labels),
        "labels": {},
    }

    for label in labels:
        profile = RSNA2023_PROFILES[label]
        direct_patterns = compile_all(profile.direct)
        assoc_patterns = compile_all(profile.associated)
        label_exclude = compile_all(profile.exclude)
        trusted_rows, counts = scan_profile(concepts, profile, global_exclude)

        direct_rows = [r for r in trusted_rows if r["tier"] == "direct"]
        assoc_rows = [r for r in trusted_rows if r["tier"] == "associated"]
        if args.max_per_tier > 0:
            direct_rows = direct_rows[: args.max_per_tier]
            assoc_rows = assoc_rows[: args.max_per_tier]
        kept_rows = [{"label": label, **row} for row in direct_rows + assoc_rows]

        label_slug = slug(label)
        write_csv(
            out_dir / f"{label_slug}.trusted_concepts.csv",
            kept_rows,
            ["label", "tier", "score", "concept_index", "concept", "matched_patterns"],
        )

        top10_rows: list[dict] = []
        top10_counts = {"direct": 0, "associated": 0, "excluded": 0, "rejected": 0}
        for rank, concept in enumerate(top10_by_label.get(label, []), start=1):
            tier, score, hits, excludes = classify_concept(
                concept, profile, global_exclude,
                direct_patterns, assoc_patterns, label_exclude,
            )
            top10_counts[tier] += 1
            top10_rows.append({
                "label": label, "rank": rank, "tier": tier, "score": score,
                "concept": concept,
                "matched_patterns": " | ".join(hits),
                "exclusion_patterns": " | ".join(excludes),
            })
        write_csv(
            out_dir / f"{label_slug}.current_top10_audit.csv",
            top10_rows,
            ["label", "rank", "tier", "score", "concept", "matched_patterns", "exclusion_patterns"],
        )

        label_auc = audit.get(label, {}).get("stats", {}).get("test_auc")
        summary = {
            "label": label,
            "auc": label_auc,
            "category": "rsna2023_trauma",
            "profile_source": profile.source,
            "profile_status": profile_status(profile, counts),
            "direct_candidates": counts["direct"],
            "associated_candidates": counts["associated"],
            "trusted_total": counts["direct"] + counts["associated"],
            "trusted_fraction": (counts["direct"] + counts["associated"]) / len(concepts),
            "top10_direct": top10_counts["direct"],
            "top10_associated": top10_counts["associated"],
            "top10_trusted": top10_counts["direct"] + top10_counts["associated"],
            "top10_excluded": top10_counts["excluded"],
            "top10_rejected": top10_counts["rejected"],
            "notes": profile.notes,
            "trusted_csv": str(out_dir / f"{label_slug}.trusted_concepts.csv"),
            "top10_audit_csv": str(out_dir / f"{label_slug}.current_top10_audit.csv"),
        }
        summary["radiology_flag"] = radiology_flag(summary)
        summary_rows.append(summary)
        manifest["labels"][label] = summary

    summary_fields = [
        "label", "auc", "category", "profile_source", "profile_status", "radiology_flag",
        "direct_candidates", "associated_candidates", "trusted_total", "trusted_fraction",
        "top10_direct", "top10_associated", "top10_trusted", "top10_excluded", "top10_rejected",
        "notes", "trusted_csv", "top10_audit_csv",
    ]
    write_csv(out_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(
        out_dir / "radiology_review_flags.csv",
        summary_rows,
        [
            "label", "auc", "category", "profile_source", "profile_status", "radiology_flag",
            "direct_candidates", "associated_candidates", "trusted_total", "top10_trusted", "notes",
        ],
    )
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"loaded {len(concepts)} concepts")
    print(f"processed {len(labels)} labels")
    print(f"wrote {out_dir}")
    for row in summary_rows:
        auc_str = f"AUC={row['auc']:.4f}" if row["auc"] is not None else "AUC=n/a"
        print(
            f"  {row['label']:<22s} {auc_str}  direct={row['direct_candidates']:>4d}  "
            f"assoc={row['associated_candidates']:>4d}  top10={row['top10_trusted']}/10  "
            f"flag={row['radiology_flag']}"
        )


if __name__ == "__main__":
    main()
