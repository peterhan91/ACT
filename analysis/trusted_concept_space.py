"""Build label-specific trusted concept spaces from the full CT concept bank.

This is a deterministic prototype for auditing whether a phenotype can be
predicted through concepts that are defensible for that label. It scans the
376k report-phrase bank, filters out obvious confounds, and separates concepts
into two tiers:

  direct      observable imaging findings that can support the label
  associated  biologically plausible but non-definitional concepts

The script also audits the current top-10 concept explanations for each label
against the trusted space.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = [
    "Pleurisy; pleural effusion",
    "Pneumonia",
    "Pulmonary congestion and hypostasis",
    "Shock",
    "Sepsis",
    "Septic shock",
    "Chronic renal failure [CKD]",
    "Obesity",
]


GLOBAL_EXCLUDE = [
    r"\b(no|without|absent|negative for|free of|not seen|not observed|not detected|not visualized|no evidence)\b",
    r"\b(normal|unremarkable|patent vasculature|appears normal|within normal)\b",
    r"\b(not optimized|limited for evaluation|suboptimal for evaluation)\b",
    r"\b(prior|previous|interval|stable|unchanged|new|newly|increased?|increasing|decreased?|decreasing|improved?|improvement|resolved?|resolution|regressed|regression|progressed?|progression|worsened?|worsening|persistent|remaining|compared)\b",
    # Plural and derived forms are listed explicitly. A bare \w* suffix is unsafe
    # here: it would make "line" match "linear", "port" match "portal" and "tube"
    # match "tuberculosis". The word boundary already protects those stems, so only
    # the missing inflections are added. 'plate' also stays singular: reports
    # write 'calcified atheroma plates' for plaque, so the plural is not procedural.
    # Deliberately NOT excluded, because each is anatomy or pathology rather than a
    # device: 'sheath' (rectus/peribronchial sheath), 'implant' (peritoneal implants
    # of carcinomatosis), 'shunt' (intrahepatic vascular shunts) and 'draining'
    # (spontaneously draining wound or sinus tract, a genuine infection finding).
    r"\b(catheters?|drains?|drainage|drained|stents?|stenting|tubes?|lines?|ports?|pacemakers?|leads?|wires?|prosthe\w*|screws?|plate|clips?|coils?|embolizations?|sutures?|staples?|stapled|stapling|tacks|\bmesh\b|anastomotic|postoperative|post-operative|postsurgical|post-surgical|post surgical|status post|s/p|resections?|anastomos[ei]s|lobectomy|pneumonectomy|whipple|hepaticojejunostomy|gastrojejunostomy|cholecystectomy|nephrectomy|transplants?|bypass|sternotomy|tracheostomy|ostomy|ileostomy|colostomy|surgical)\b",
    r"\b(possible|possibly|probable|probably|suggest|suggesting|suggestive|compatible|consistent|likely|may represent|could represent|cannot exclude|suspected|suspicious|concerning|favoring|favored|recommend|follow[- ]up)\b",
    r"\b(history of|known|diagnosed|diagnosis|treated|therapy|chemotherapy|radiotherapy)\b",
    # Negated/absent findings missed by the phrase-specific negation above.
    # Only negations that ALWAYS mean the finding is absent are global here; tokens
    # like "non-obstructing"/"non-dilated" describe a sub-attribute of a real finding
    # (e.g. "hernia containing non-dilated bowel") so they are handled LOCALLY in the
    # AKI (non-obstruct) and ileus (non-dilat) profiles, not globally.
    r"\bnon[- ]?(aneurysm\w*|enlarg\w*)\b",
    r"\bnot (aneurysm\w*|enlarged|collapsed)\b",
    # 'emphysema' that is soft-tissue air or a visceral gas infection, never pulmonary:
    r"\b(subcutaneous|soft[- ]tissue|surgical) emphysema\b",
    r"\bemphysematous (cholecystitis|cystitis|gastritis|pyelonephritis|osteomyelitis)\b",
]


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class LabelProfile:
    label: str
    direct: tuple[str, ...]
    associated: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    notes: str = ""
    source: str = "curated"


PROFILES = {
    "Pleurisy; pleural effusion": LabelProfile(
        label="Pleurisy; pleural effusion",
        direct=(
            r"\bpleural effusions?\b",
            r"\bpleural fluid\b",
            r"\bpleural collection\b",
            r"\bhydropneumothorax\b",
            r"\bfluid in (the )?pleural\b",
        ),
        exclude=(r"\bpericardial\b", r"\bascites\b", r"\bperitoneal\b"),
        notes="Pleural fluid/effusion terms are direct findings.",
    ),
    "Pneumonia": LabelProfile(
        label="Pneumonia",
        direct=(
            r"\bconsolidat\w*\b",
            r"\bair bronchograms?\b",
            r"\bground[- ]glass\b.*\bconsolidat\w*\b",
            r"\bconsolidat\w*\b.*\bground[- ]glass\b",
            r"\b(tree[- ]in[- ]bud|centrilobular nodules?)\b",
            r"\bpneumonic infiltrat\w*\b",
            r"\bparapneumonic effusion\b",
        ),
        associated=(
            r"\bground[- ]glass\b",
            r"\bpleural effusions?\b",
            r"\bbronchial wall thickening\b",
        ),
        exclude=(r"\bpancrea", r"\bbile duct\b", r"\bbiliary\b"),
        notes="Direct tier captures observable air-space infection patterns.",
    ),
    "Pulmonary congestion and hypostasis": LabelProfile(
        label="Pulmonary congestion and hypostasis",
        direct=(
            r"\bpulmonary (edema|oedema|congestion)\b",
            r"\bsmooth septal thickening\b",
            r"\bperihilar edema\b",
            r"\bground[- ]glass.*edema\b",
        ),
        associated=(
            r"\binterlobular septal thickening\b",
            r"\b(lung|pulmonary|perihilar|hilar).*\bvascular congestion\b",
            r"\bpleural effusions?\b",
            r"\bpericardial effusions?\b",
            r"\bivc\b.*\breflux\b",
            r"\bhepatic venous\b.*\breflux\b",
            r"\bhepatic veins?\b.*\bcongestion\b",
        ),
        exclude=(r"\bbrain\b", r"\bpancrea", r"\bbile duct\b", r"\bbiliary\b"),
        notes="IVC/hepatic reflux is associated, not direct pulmonary edema.",
    ),
    "Sepsis": LabelProfile(
        label="Sepsis",
        direct=(),
        associated=(
            r"\b(abscess|infected collection|empyema)\b",
            r"\bpneumonia\b",
            r"\bpyelonephritis\b",
            r"\bcolitis\b",
            r"\bdiverticulitis\b",
            r"\bcholangitis\b",
            r"\bperitonitis\b",
            r"\binfectious\b",
        ),
        exclude=(r"\bivc\b", r"\bhepatic venous\b", r"\batheroscl", r"\bcalcif"),
        notes="Sepsis has no direct CT-definitional sign; only source/injury associations are allowed.",
    ),
    "Shock": LabelProfile(
        label="Shock",
        direct=(
            r"\bshock\b",
            r"\b(systemic|global) hypoperfusion\b",
            r"\bbilateral renal hypoperfusion\b",
            r"\bhypoperfusion of both kidneys\b",
            r"\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b.*\bivc\b",
            r"\bivc\b.*\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b",
            r"\bsmall caliber\b.*\baorta\b",
        ),
        associated=(
            r"\badrenal\w*.*\bhyperenhanc\w*\b",
            r"\b(tamponade|large pericardial effusion|moderate pericardial effusion)\b",
            r"\bactive (contrast )?extravasation\b",
            r"\bhemoperitoneum\b",
            r"\b(abscess|infected collection|empyema|pyelonephritis|cholangitis|peritonitis)\b",
        ),
        exclude=(
            r"\bperihepatic effusions?\b",
            r"\b(distended|engorged|prominent|enlarged) (ivc|hepatic veins?)\b",
            r"\b(ivc|hepatic veins?).*\b(distended|engorged|prominent|enlarged)\b",
            r"\b(ivc|hepatic veins?).*\breflux\b",
            r"\breflux\b.*\b(ivc|hepatic veins?)\b",
            r"\bpatent hepatic vasculature\b",
            r"\bmass effect\b",
            r"\bcompression\b",
        ),
        notes="Clean shock concepts are hypoperfusion/low-volume physiology; venous reflux and perihepatic fluid are fluid/pressure confounds.",
    ),
    "Septic shock": LabelProfile(
        label="Septic shock",
        direct=(),
        associated=(
            r"\bshock\b",
            r"\b(systemic|global) hypoperfusion\b",
            r"\bbilateral renal hypoperfusion\b",
            r"\bhypoperfusion of both kidneys\b",
            r"\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b.*\bivc\b",
            r"\bivc\b.*\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b",
            r"\bsmall caliber\b.*\baorta\b",
            r"\badrenal\w*.*\bhyperenhanc\w*\b",
            r"\b(abscess|infected collection|empyema|pyelonephritis|cholangitis|peritonitis|pneumonia)\b",
        ),
        exclude=(
            r"\bperihepatic effusions?\b",
            r"\b(distended|engorged|prominent|enlarged) (ivc|hepatic veins?)\b",
            r"\b(ivc|hepatic veins?).*\b(distended|engorged|prominent|enlarged)\b",
            r"\b(ivc|hepatic veins?).*\breflux\b",
            r"\breflux\b.*\b(ivc|hepatic veins?)\b",
            r"\bpatent hepatic vasculature\b",
            r"\bmass effect\b",
            r"\bcompression\b",
        ),
        notes="Septic shock is a clinical state; CT can supply infection source or shock-physiology associations, not direct diagnosis.",
    ),
    "Chronic renal failure [CKD]": LabelProfile(
        label="Chronic renal failure [CKD]",
        direct=(
            r"\b(renal|kidney|kidneys)\b.{0,40}\b(atroph\w*|cortical thinning|cortical scarring)\b",
            r"\b(atroph\w*)\b.{0,40}\b(renal|kidney|kidneys)\b",
            r"\bsmall (bilateral )?(kidneys|renal size)\b",
            r"\b(kidneys|renal size) (are|is|appear|appears)? ?small\b",
            r"\bbilateral renal cortical thinning\b",
        ),
        associated=(
            r"\bvascular calcification\b",
            r"\baortic calcif\w*\b",
            r"\bcoronary (artery )?calcif\w*\b",
            r"\bcalcified atheroma\w*\b",
            r"\batherosclerotic calcif\w*\b",
        ),
        exclude=(
            r"\bhydronephrosis\b",
            r"\bacute\b",
            r"\btoo small to characterize\b",
            r"\bsmall foci\b",
            r"\bsmall amount\b",
            r"\blesion\b",
        ),
        notes="Vascular calcification is plausible CKD-MBD association, not direct renal-failure imaging.",
    ),
    "Obesity": LabelProfile(
        label="Obesity",
        direct=(
            r"\bincreased visceral (adipose|fat)\b",
            r"\bexcess(ive)? visceral (adipose|fat)\b",
            r"\bobese body habitus\b",
            r"\blarge body habitus\b",
        ),
        associated=(
            r"\bhepatic steatosis\b",
            r"\bfatty liver\b",
            r"\bvascular calcification\b",
            r"\baortic calcif\w*\b",
            r"\bcoronary (artery )?calcif\w*\b",
            r"\bcalcified atheroma\w*\b",
            r"\batherosclerotic calcif\w*\b",
        ),
        notes="Obesity is usually a weak CT-report target; direct concepts are body-fat/body-habitus terms only.",
    ),
}


PLEURAL_DIRECT = (
    r"\bpleural effusions?\b",
    r"\bpleural fluid\b",
    r"\bpleural collection\b",
    r"\bhydropneumothorax\b",
    r"\bfluid in (the )?pleural\b",
)
PNEUMONIA_DIRECT = (
    r"\bconsolidat\w*\b",
    r"\bair bronchograms?\b",
    r"\bground[- ]glass\b.*\bconsolidat\w*\b",
    r"\bconsolidat\w*\b.*\bground[- ]glass\b",
    r"\b(tree[- ]in[- ]bud|centrilobular nodules?)\b",
    r"\bpneumonic infiltrat\w*\b",
    r"\bparapneumonic effusion\b",
)
PNEUMONIA_ASSOC = (
    r"\bground[- ]glass\b",
    r"\bpleural effusions?\b",
    r"\bbronchial wall thickening\b",
)
PULMONARY_EDEMA_DIRECT = (
    r"\bpulmonary (edema|oedema|congestion)\b",
    r"\bsmooth septal thickening\b",
    r"\bperihilar edema\b",
    r"\bground[- ]glass.*edema\b",
)
PULMONARY_EDEMA_ASSOC = (
    r"\binterlobular septal thickening\b",
    r"\b(lung|pulmonary|perihilar|hilar).*\bvascular congestion\b",
)
LUNG_CANCER_DIRECT = (
    r"\b(lung|pulmonary|bronchial|bronchus).*\b(mass|neoplasm|tumou?r|malignan\w*|metasta\w*)\b",
    r"\b(mass|neoplasm|tumou?r|malignan\w*|metasta\w*).*\b(lung|pulmonary|bronchial|bronchus)\b",
    r"\b(metastatic|malignant|spiculated).*\b(lung|pulmonary).*\bnodules?\b",
    r"\b(lung|pulmonary).*\bnodules?\b.*\b(metastatic|malignant|spiculated)\b",
)
LUNG_CANCER_ASSOC = (
    r"\b(lung|pulmonary).*\bnodules?\b",
    r"\bnodules?.*\b(lung|pulmonary)\b",
    r"\blymphadenopathy\b",
    r"\bpleural effusions?\b",
)
VASCULAR_CALC_ASSOC = (
    r"\bvascular calcification\b",
    r"\baortic calcif\w*\b",
    r"\bcoronary (artery )?calcif\w*\b",
    r"\bcalcified atheroma\w*\b",
    r"\batherosclerotic calcif\w*\b",
)
CKD_DIRECT = (
    r"\b(renal|kidney|kidneys)\b.{0,40}\b(atroph\w*|cortical thinning|cortical scarring)\b",
    r"\b(atroph\w*)\b.{0,40}\b(renal|kidney|kidneys)\b",
    r"\bsmall (bilateral )?(kidneys|renal size)\b",
    r"\b(kidneys|renal size) (are|is|appear|appears)? ?small\b",
    r"\bbilateral renal cortical thinning\b",
)
OBESITY_DIRECT = (
    r"\bincreased visceral (adipose|fat)\b",
    r"\bexcess(ive)? visceral (adipose|fat)\b",
    r"\bobese body habitus\b",
    r"\blarge body habitus\b",
)
OBESITY_ASSOC = (
    r"\bhepatic steatosis\b",
    r"\bfatty liver\b",
) + VASCULAR_CALC_ASSOC
SHOCK_LOW_VOLUME = (
    r"\bshock\b",
    r"\b(systemic|global) hypoperfusion\b",
    r"\bbilateral renal hypoperfusion\b",
    r"\bhypoperfusion of both kidneys\b",
    r"\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b.*\bivc\b",
    r"\bivc\b.*\b(collapsed|flattened|flat|slit[- ]?like|small caliber)\b",
    r"\bsmall caliber\b.*\baorta\b",
)
SHOCK_ASSOC = (
    r"\badrenal\w*.*\bhyperenhanc\w*\b",
)
SHOCK_EXCLUDE = (
    r"\bperihepatic effusions?\b",
    r"\b(distended|engorged|prominent|enlarged) (ivc|hepatic veins?)\b",
    r"\b(ivc|hepatic veins?).*\b(distended|engorged|prominent|enlarged)\b",
    r"\b(ivc|hepatic veins?).*\breflux\b",
    r"\breflux\b.*\b(ivc|hepatic veins?)\b",
    r"\bpatent hepatic vasculature\b",
    r"\bmass effect\b",
    r"\bcompression\b",
)


def make_profile(
    label: str,
    *,
    direct: tuple[str, ...] = (),
    associated: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    notes: str,
    source: str,
) -> LabelProfile:
    return LabelProfile(
        label=label,
        direct=direct,
        associated=associated,
        exclude=exclude,
        notes=notes,
        source=source,
    )


def auto_profile(label: str, category: str = "") -> LabelProfile:
    """Conservative fallback profiles for the full INSPECT label set.

    Empty profiles are intentional for clinical labels that do not have a
    defensible CT-report concept space.
    """
    l = label.lower()

    if label in PROFILES:
        return PROFILES[label]
    if "pleural" in l and "effusion" in l:
        return make_profile(
            label, direct=PLEURAL_DIRECT,
            exclude=(r"\bpericardial\b", r"\bascites\b", r"\bperitoneal\b"),
            notes="Auto direct profile from pleural-effusion label text.",
            source="auto_label",
        )
    if "pneumonia" in l or "pneumonitis" in l:
        return make_profile(
            label, direct=PNEUMONIA_DIRECT,
            associated=PNEUMONIA_ASSOC,
            exclude=(r"\bpancrea", r"\bbile duct\b", r"\bbiliary\b"),
            notes="Auto pneumonia profile from label text/category.",
            source="auto_label",
        )
    if "septic shock" in l:
        return PROFILES["Septic shock"]
    if l == "shock":
        return PROFILES["Shock"]
    if "sepsis" in l:
        return PROFILES["Sepsis"]
    if "systemic inflammatory response" in l or l == "sirs":
        return make_profile(
            label,
            associated=PROFILES["Sepsis"].associated,
            exclude=PROFILES["Sepsis"].exclude,
            notes="SIRS is clinical; CT can only provide infection/source associations.",
            source="auto_label",
        )
    if "dependence on respirator" in l or "ventilator" in l or "supplemental oxygen" in l:
        return make_profile(
            label,
            notes="Ventilator/oxygen dependence is a care-status label, not a trustworthy CT concept target.",
            source="radiology_review_empty",
        )
    if "hypotension" in l or "hypovolemia" in l:
        return make_profile(
            label, associated=SHOCK_LOW_VOLUME + SHOCK_ASSOC, exclude=SHOCK_EXCLUDE,
            notes="Clinical low-pressure/volume label; CT can only provide associated low-volume physiology.",
            source="auto_label",
        )
    if "pulmonary embolism" in l:
        return make_profile(
            label,
            direct=(
                r"\bpulmonary embol\w*\b",
                r"\bembol\w*.*\bpulmonary arter\w*\b",
                r"\bpulmonary arter\w*.*\b(filling defect|embol\w*)\b",
            ),
            notes="Auto direct profile from pulmonary-embolism label text.",
            source="auto_label",
        )
    if "pulmonary nodule" in l or "lung nodule" in l:
        return make_profile(
            label,
            direct=(r"\b(lung|pulmonary).*\bnodules?\b", r"\bnodules?.*\b(lung|pulmonary)\b"),
            exclude=(r"\blymph\b", r"\bthyroid\b", r"\badrenal\b"),
            notes="Auto direct profile from pulmonary-nodule label text.",
            source="auto_label",
        )
    if "pneumothorax" in l:
        return make_profile(
            label,
            direct=(r"\bpneumothorax\b", r"\bhydropneumothorax\b"),
            associated=(r"\bempyema\b", r"\bpleural effusions?\b"),
            notes="Auto direct pneumothorax profile from label text.",
            source="auto_label",
        )
    if "ascites" in l or category == "GI_ASCITES":
        return make_profile(
            label,
            direct=(
                r"\bascites\b",
                r"\bintraperitoneal fluid\b",
                r"\bperitoneal fluid\b",
                r"\bintra[- ]abdominal (effusion|fluid)\b",
                r"\bfree fluid\b",
            ),
            exclude=(r"\bpleural\b", r"\bpericardial\b"),
            notes="Auto direct ascites/peritoneal-fluid profile.",
            source="auto_label" if "ascites" in l else "auto_category",
        )
    if "ileus" in l or category == "GI_ILEUS":
        return make_profile(
            label,
            direct=(
                r"\bileus\b",
                r"\bbowel disten\w*\b",
                r"\bdilated (small |large )?bowel\b",
                r"\bair[- ]fluid levels?\b",
                r"\bfluid[- ]filled bowel\b",
            ),
            exclude=(
                # 'air-fluid level' also occurs in extraluminal/non-bowel containers;
                # those are not ileus. Negated "non-dilated bowel" = absent, not ileus.
                r"\b(stomach|gastric|cyst|pseudocyst|abscess|bladder|pleural|hydropneumothorax)\b",
                r"\bnon[ -]?dilat\w*\b",
            ),
            notes="Auto direct ileus/bowel-distension profile; extraluminal air-fluid "
                  "(gastric/cyst/abscess/bladder/pleural) excluded.",
            source="auto_label" if "ileus" in l else "auto_category",
        )
    if "hiatal hernia" in l or "diaphragmatic hernia" in l or category == "DIRECT_HIATAL_HERNIA":
        if "abdominal hernia" in l:
            hernia_direct = (
                r"\babdominal hernia\b",
                r"\bventral hernia\b",
                r"\bumbilical hernia\b",
                r"\bincisional hernia\b",
                r"\binguinal hernia\b",
                r"\bfemoral hernia\b",
            )
        else:
            hernia_direct = (r"\bhiatal hernia\b", r"\bdiaphragmatic hernia\b")
        return make_profile(
            label,
            direct=hernia_direct,
            notes="Auto direct hernia profile.",
            source="auto_label" if "hernia" in l else "auto_category",
        )
    if "coronary atherosclerosis" in l or category == "DIRECT_CORONARY_CALC":
        return make_profile(
            label,
            direct=(
                r"\bcoronary (artery )?(calcif\w*|atheroscl\w*|atheroma\w*)\b",
                r"\b(calcif\w*|atheroscl\w*|atheroma\w*).*\bcoronary\b",
            ),
            associated=(r"\baortic calcif\w*\b", r"\batherosclerotic calcif\w*\b"),
            notes="Auto direct coronary calcification/atherosclerosis profile.",
            source="auto_label" if "coronary" in l else "auto_category",
        )
    if "cardiomegaly" in l or category == "DIRECT_CARDIOMEGALY":
        return make_profile(
            label,
            direct=(r"\bcardiomegaly\b", r"\benlarged (heart|cardiac silhouette)\b", r"\bcardiac enlargement\b"),
            notes="Auto direct cardiomegaly profile.",
            source="auto_label" if "cardiomegaly" in l else "auto_category",
        )
    if "pericarditis" in l or category == "DIRECT_PERICARDIAL":
        return make_profile(
            label,
            direct=(r"\bpericarditis\b", r"\bpericardial thickening\b", r"\bpericardial enhancement\b"),
            associated=(r"\bpericardial effusions?\b",),
            notes="Pericarditis is usually clinical; pericardial inflammation is direct, effusion associated.",
            source="auto_label" if "pericarditis" in l else "auto_category",
        )
    if ("emphysema" in l and "pulmonary collapse" not in l) or category == "DIRECT_EMPHYSEMA":
        return make_profile(
            label,
            direct=(r"\bemphysema\b", r"\bemphysematous\b", r"\bhyperinflation\b", r"\bbullous\b"),
            notes="Auto direct emphysema profile.",
            source="auto_label" if "emphysema" in l else "auto_category",
        )
    if "bronchiectasis" in l or category == "BRONCHIECTASIS":
        return make_profile(
            label,
            direct=(r"\bbronchiect\w*\b", r"\bbronchial dilatation\b"),
            associated=(r"\bbronchial wall thickening\b",),
            notes="Auto direct bronchiectasis profile.",
            source="auto_label" if "bronchiectasis" in l else "auto_category",
        )
    if "bronchitis" in l or category == "BRONCHITIS":
        return make_profile(
            label,
            associated=(r"\bbronchial wall thickening\b", r"\bperibronchial thickening\b", r"\bbronchiect\w*\b", r"\bemphysema\b"),
            notes="Chronic bronchitis/COPD labels have associated airway concepts but are not direct CT diagnoses.",
            source="auto_label" if "bronchitis" in l else "auto_category",
        )
    if "pulmonary collapse" in l or "atelectasis" in l or category == "PULMONARY_COLLAPSE":
        return make_profile(
            label,
            direct=(
                r"\batelecta\w*\b",
                r"\bpulmonary collapse\b",
                r"\blobar collapse\b",
                r"\bvolume loss\b",
                r"\bemphysema\b",
                r"\bemphysematous\b",
            ),
            notes="Auto direct pulmonary-collapse/atelectasis profile.",
            source="auto_label" if ("collapse" in l or "atelectasis" in l) else "auto_category",
        )
    if "pulmonary congestion" in l or "pulmonary edema" in l or category == "PULMONARY_EDEMA_CONGESTION":
        return make_profile(
            label,
            direct=PULMONARY_EDEMA_DIRECT,
            associated=PULMONARY_EDEMA_ASSOC + (
                r"\bpleural effusions?\b",
                r"\bpericardial effusions?\b",
                r"\bivc\b.*\breflux\b",
                r"\bhepatic venous\b.*\breflux\b",
                r"\bhepatic veins?\b.*\bcongestion\b",
            ),
            exclude=(r"\bbrain\b", r"\bpancrea", r"\bbile duct\b", r"\bbiliary\b"),
            notes="Auto pulmonary edema/congestion profile; venous reflux remains associated only.",
            source="auto_label" if ("congestion" in l or "edema" in l) else "auto_category",
        )
    if "respiratory failure" in l or "respiratory insufficiency" in l or category == "RESP_FAILURE":
        return make_profile(
            label,
            associated=PNEUMONIA_DIRECT + PNEUMONIA_ASSOC + PULMONARY_EDEMA_DIRECT + PULMONARY_EDEMA_ASSOC + PLEURAL_DIRECT + (
                r"\batelecta\w*\b",
                r"\bpneumothorax\b",
            ),
            notes="Respiratory failure is clinical; CT findings are associated causes/consequences only.",
            source="auto_label" if "respiratory" in l else "auto_category",
        )
    if "heart failure" in l or category == "HEART_FAILURE":
        return make_profile(
            label,
            associated=PULMONARY_EDEMA_DIRECT + PULMONARY_EDEMA_ASSOC + PLEURAL_DIRECT + (
                r"\bcardiomegaly\b",
                r"\bivc\b.*\breflux\b",
                r"\bhepatic venous\b.*\breflux\b",
            ),
            notes="Heart failure is clinical; CT signs are associated congestion findings.",
            source="auto_label" if "heart failure" in l else "auto_category",
        )
    if "chronic renal" in l or "chronic kidney" in l or "end stage renal" in l or category == "CKD":
        return make_profile(
            label,
            direct=CKD_DIRECT,
            associated=VASCULAR_CALC_ASSOC,
            exclude=(
                r"\bhydronephrosis\b",
                r"\bacute\b",
                r"\btoo small to characterize\b",
                r"\bsmall foci\b",
                r"\bsmall amount\b",
                r"\blesion\b",
            ),
            notes="Auto CKD profile; vascular calcification is associated, renal atrophy/scarring direct.",
            source="auto_label" if ("renal" in l or "kidney" in l) else "auto_category",
        )
    if "diabetes" in l or category == "T2D_RENAL":
        return make_profile(
            label,
            associated=VASCULAR_CALC_ASSOC + (r"\bfatty liver\b", r"\bhepatic steatosis\b"),
            notes="Diabetes has no direct CT diagnosis; retained concepts are vascular/metabolic associations.",
            source="auto_label" if "diabetes" in l else "auto_category",
        )
    if "obesity" in l or "hyperalimentation" in l or category == "OBESITY":
        return make_profile(
            label, direct=OBESITY_DIRECT, associated=OBESITY_ASSOC,
            notes="Auto obesity profile; body-fat/habitus direct, steatosis/calcification associated.",
            source="auto_label" if ("obesity" in l or "hyperalimentation" in l) else "auto_category",
        )
    if (
        category == "LUNG_CANCER"
        or ("cancer" in l and ("lung" in l or "bronchus" in l or "respiratory" in l))
        or ("malignan" in l and "respiratory" in l)
    ):
        return make_profile(
            label,
            direct=LUNG_CANCER_DIRECT,
            associated=LUNG_CANCER_ASSOC,
            notes="Auto lung-cancer profile from label text/category.",
            source="auto_label" if ("lung" in l or "bronchus" in l or "respiratory" in l) else "auto_category",
        )
    if "secondary malignancy of lymph nodes" in l:
        return make_profile(
            label,
            direct=(
                r"\bmetasta\w*.*\blymph(adenopathy| nodes?)\b",
                r"\blymph(adenopathy| nodes?).*\bmetasta\w*\b",
                r"\bmalignant.*\blymph nodes?\b",
            ),
            associated=(r"\blymphadenopathy\b", r"\benlarged lymph nodes?\b"),
            notes="Lymph-node metastasis requires malignant/metastatic wording; generic lymphadenopathy is associated only.",
            source="auto_label",
        )
    if "secondary malignancy of bone" in l:
        return make_profile(
            label,
            direct=(r"\b(osseous|bone|skeletal).*\b(metasta\w*|lytic|sclerotic)\b", r"\b(metasta\w*|lytic|sclerotic).*\b(osseous|bone|skeletal)\b"),
            exclude=(r"\bbone isl\w*", r"\benostos\w*\b", r"\bhemangioma\b",
                     r"\bbenign\b", r"\bbone infarct\b", r"\bdiscogenic\b"),
            notes="Auto direct osseous-metastasis profile; benign sclerotic mimics "
                  "(bone islands / enostoses / hemangiomas) excluded.",
            source="auto_label",
        )
    if category == "CANCER_GENERAL":
        return make_profile(
            label,
            associated=(r"\bmetasta\w*\b", r"\blymphadenopathy\b", r"\bneoplasm\b", r"\btumou?r\b"),
            notes="Broad cancer label; retained only as associated malignancy concepts.",
            source="auto_category",
        )
    if "aneurysm" in l:
        return make_profile(
            label,
            direct=(
                r"\baneurysm\w*\b",
                r"\b(aorta|aortic|artery|arterial|iliac|visceral branch\w*).*\b(ectasia|ectatic)\b",
                r"\b(ectasia|ectatic).*\b(aorta|aortic|artery|arterial|iliac|visceral branch\w*)\b",
            ),
            notes="Auto direct aneurysm profile.",
            source="auto_label",
        )
    if "deep vein thrombosis" in l or "dvt" in l:
        return make_profile(
            label,
            direct=(
                r"\bdeep vein thrombosis\b",
                r"\bdvt\b",
                r"\b(common femoral|femoral|external iliac|common iliac|popliteal|lower extremity).*\b(thromb\w*|filling defect)\b",
                r"\b(thromb\w*|filling defect).*\b(common femoral|femoral|external iliac|common iliac|popliteal|lower extremity)\b",
            ),
            exclude=(r"\barter", r"\baneurysm", r"\bpseudoaneurysm", r"\bportal\b", r"\bhepatic\b", r"\bovarian\b", r"\bmesenteric\b", r"\bivc\b"),
            notes="DVT profile restricted to lower-extremity/deep venous thrombosis concepts.",
            source="auto_label",
        )
    if "venous embolism" in l:
        return make_profile(
            label,
            direct=(r"\b(deep vein thrombosis|dvt|venous thromb\w*)\b", r"\bvenous.*\bfilling defect\b"),
            notes="Auto direct venous thrombosis profile from label text.",
            source="auto_label",
        )
    if "urinary tract infection" in l:
        return make_profile(
            label,
            associated=(r"\bpyelonephritis\b", r"\bcystitis\b", r"\burosepsis\b"),
            notes="UTI is clinical; CT concepts are associated source findings.",
            source="auto_label",
        )
    if "abscess" in l:
        return make_profile(
            label,
            direct=(r"\babscess\b", r"\brim[- ]enhancing (fluid )?collection\b", r"\binfected collection\b"),
            exclude=(
                # 'Superficial cellulitis and abscess' is a soft-tissue label; deep
                # visceral / intra-cavitary abscesses are a different entity.
                r"\b(hepatic|liver|splenic|spleen|renal|perinephric|perirenal|(ilio)?psoas|"
                r"retroperitoneal|intra[- ]?abdominal|intraperitoneal|pelvic|presacral|"
                r"peritoneal|mesenteric|tubo[- ]?ovarian|adnexal|prostat\w*|pancreat\w*|"
                r"subphrenic|subhepatic|perihepatic|perisplenic|paracolic|diverticular|"
                r"appendiceal|periappendiceal|colon|colonic|duodenum|lung|pulmonary|"
                r"mediastinal|epidural|intracranial|cerebral)\b",
            ),
            notes="Superficial soft-tissue abscess profile; deep visceral/intra-cavitary "
                  "abscesses excluded.",
            source="auto_label",
        )
    if "cirrhosis" in l:
        return make_profile(
            label,
            direct=(r"\bcirrhosis\b", r"\bcirrhotic\b", r"\bnodular liver\b"),
            associated=(r"\bascites\b", r"\bsplenomegaly\b", r"\bvarices\b"),
            notes="Auto cirrhosis profile from label text.",
            source="auto_label",
        )
    if "osteoporosis" in l or category == "OSTEOPOROSIS":
        return make_profile(
            label,
            associated=(r"\bosteopenia\b", r"\bosteopor\w*\b", r"\bcompression fracture\b", r"\bvertebral fracture\b"),
            notes="Osteoporosis is not a direct CT diagnosis here; retained bone-density/fracture associations.",
            source="auto_label" if "osteoporosis" in l else "auto_category",
        )
    if category == "HEART_VALVE":
        return make_profile(
            label,
            associated=(r"\bvalvular calcif\w*\b", r"\baortic valve calcif\w*\b", r"\bmitral annular calcif\w*\b"),
            notes="Valve-disease label; CT concepts are calcification associations.",
            source="auto_category",
        )

    # ------------------------------------------------------------------
    # Radiologist-review additions (2026-06): phenotypes that DO have a
    # defensible CT concept basis but previously fell through to the empty
    # default because no profile branch matched their label text. These are
    # appended AFTER every existing branch so they can only convert
    # previously-empty (dropped) labels — no kept/handled label changes path.
    # ------------------------------------------------------------------
    if "pulmonary heart disease" in l or "cor pulmonale" in l or category == "COR_PULMONALE":
        return make_profile(
            label,
            direct=(
                r"\bcor pulmonale\b",
                r"\bright (ventric\w*|atri\w*|heart)\b.{0,40}\b(enlarg\w*|dilat\w*|strain|prominent)\b",
                r"\b(enlarg\w*|dilat\w*|strain|prominent)\b.{0,40}\bright (ventric\w*|atri\w*|heart)\b",
                r"\brv[:/ ]?lv ratio\b",
                r"\b(main )?pulmonary arter\w*\b.{0,40}\b(enlarg\w*|dilat\w*|prominent)\b",
                r"\b(enlarg\w*|dilat\w*|prominent)\b.{0,40}\b(main )?pulmonary arter\w*\b",
                r"\b(interventricular )?septal (bowing|flattening|deviation|bulging)\b",
                r"\bpulmonary (arterial )?hypertension\b",
            ),
            associated=(
                r"\bemphysema\w*\b",
                r"\bbullous\b",
                r"\bright heart failure\b",
            ),
            exclude=(
                # bare 'emphysema' also matches non-pulmonary air — keep only lung emphysema
                r"\bsubcutaneous emphysema\b",
                r"\bsoft[ -]tissue emphysema\b",
                r"\bsurgical emphysema\b",
                r"\bemphysematous (cholecystitis|cystitis|gastritis|pyelonephritis|osteomyelitis)\b",
            ),
            notes=(
                "Cor pulmonale / pulmonary heart disease: direct tier is the CT "
                "morphology of right-heart strain (RV/RA and pulmonary-artery "
                "enlargement, RV:LV ratio, septal bowing, pulmonary hypertension). "
                "Acute PE is deliberately NOT included to avoid circularity on a PE "
                "dataset; chronic airway substrate is associated only."
            ),
            source="auto_label",
        )
    if "osteoarthr" in l or "spondylosis" in l or "degenerative joint" in l or category == "OSTEOARTHRITIS":
        return make_profile(
            label,
            direct=(
                r"\bosteophyt\w*\b",
                r"\bdegenerative (change\w*|disc disease|disc|spondyl\w*|arthropathy)\b",
                r"\bfacet (arthropathy|hypertrophy|degenerati\w*|osteoarthr\w*)\b",
                r"\b(disc|disk|intervertebral) (space )?(narrowing|height loss)\b",
                r"\bspondyl(osis|olisthesis|otic)\b",
                r"\bendplate (sclerosis|osteophyt\w*|degenerati\w*)\b",
                r"\bjoint space narrowing\b",
            ),
            notes=(
                "Osteoarthrosis / degenerative skeletal disease: degenerative spine "
                "and joint findings (osteophytes, facet arthropathy, disc-space "
                "narrowing, spondylosis) are directly visible on chest CT."
            ),
            source="auto_label",
        )
    if "musculoskeletal" in l and ("abnormal" in l or "nonspecific" in l):
        return make_profile(
            label,
            associated=(
                r"\bosteophyt\w*\b",
                r"\bdegenerative (change\w*|disc disease|spondyl\w*|arthropathy)\b",
                r"\bfacet (arthropathy|hypertrophy)\b",
                r"\bcompression fracture\b",
                r"\bvertebral (body )?(fracture|height loss)\b",
            ),
            notes=(
                "Nonspecific MSK abnormal findings: no single direct diagnosis; "
                "retained common degenerative/fracture findings as associated only."
            ),
            source="auto_label",
        )
    if "nonalcoholic liver" in l or "disorders of liver" in l or category == "NAFLD":
        return make_profile(
            label,
            direct=(
                r"\bhepatic steatos\w*\b",
                r"\bfatty liver\b",
                r"\bfatty infiltration of the liver\b",
                r"\b(liver|hepatic)\b.{0,20}\bsteatos\w*\b",
                r"\bsteatos\w*\b.{0,20}\b(liver|hepatic)\b",
            ),
            associated=(
                r"\bnodular (liver|hepatic) (contour|surface|margin)\b",
                r"\bhepatic fibrosis\b",
                r"\bhepatomegaly\b",
            ),
            notes=(
                "Chronic non-alcoholic liver disease: hepatic steatosis / fatty liver "
                "is a direct, CT-quantifiable finding; nodular contour / fibrosis / "
                "hepatomegaly are associated. Focal hepatic lesions are intentionally "
                "not matched."
            ),
            source="auto_label",
        )
    if "acute renal failure" in l or "acute kidney injury" in l or category == "AKI":
        return make_profile(
            label,
            direct=(
                r"\bhydronephros\w*\b",
                r"\bhydroureter\w*\b",
                r"\bpelvicaliectasis\b",
                r"\b(calic|caly)ectasis\b",
                r"\bobstructing\b.{0,30}\b(calcul\w*|stone|ureter\w*)\b",
                r"\b(ureter\w*|renal|kidney)\b.{0,30}\bobstruct\w*\b",
                r"\bperinephric (strand\w*|fluid|fat strand\w*)\b",
            ),
            associated=(
                r"\benlarged (kidney|kidneys|renal)\b",
                r"\b(kidneys?) (are |is |appear\w* |diffusely |bilaterally )*enlarg\w*\b",
                r"\b(bilateral )?renal enlarg\w*\b",
            ),
            exclude=(
                r"\bnon[ -]?obstruct\w*\b",   # negation: "non obstructing" stones are incidental, not AKI
                r"\bchronic\b",
                r"\batroph\w*\b",
                r"\bpolycystic\b",            # ADPKD = chronic, not acute
                r"\bbile duct\b",             # biliary obstruction is wrong organ
                r"\bbiliary\b",
                r"\bcommon bile\b",
            ),
            notes=(
                "Acute renal failure: defensible direct signs are obstructive uropathy "
                "(hydronephrosis / hydroureter / obstructing calculus / perinephric "
                "stranding). Non-obstructing stones and chronic atrophic morphology "
                "are excluded (the latter belongs to CKD)."
            ),
            source="auto_label",
        )
    if "pancreas" in l or "pancreatitis" in l or category == "PANCREATITIS":
        return make_profile(
            label,
            direct=(
                r"\bpancreatitis\b",
                r"\bperipancreatic (strand\w*|fluid|inflam\w*|edema|fat strand\w*)\b",
                r"\bpancrea\w* (necros\w*|edema|enlarge\w*|inflam\w*|swelling)\b",
            ),
            associated=(
                r"\bperipancreatic fluid collection\b",
                r"\b(pancreatic )?pseudocyst\b",
                r"\bpancreatic duct\w*\b.{0,20}\bdilat\w*\b",
                r"\bpancreatic (atrophy|calcif\w*)\b",
            ),
            notes=(
                "Diseases of pancreas: acute pancreatitis findings (peripancreatic "
                "stranding / fluid / necrosis, pancreatic edema) are direct; ductal "
                "dilatation, pseudocyst and chronic atrophy/calcification are "
                "associated. Neoplastic pancreatic disease is not claimed as direct."
            ),
            source="auto_label",
        )
    if l == "edema" or "swelling of limb" in l or "localized edema" in l:
        return make_profile(
            label,
            associated=(
                r"\bsubcutaneous (edema|stranding)\b",
                r"\banasarca\b",
                r"\bsoft[- ]tissue (edema|swelling)\b",
                r"\bperipheral edema\b",
                r"\bbody wall edema\b",
            ),
            notes=(
                "Generalized/limb edema is non-specific; CT can show subcutaneous "
                "edema / anasarca / soft-tissue swelling as associated findings only."
            ),
            source="auto_label",
        )
    if "asthma" in l:
        return make_profile(
            label,
            associated=(
                r"\bbronchial wall thickening\b",
                r"\bperibronchial thickening\b",
                r"\bair[- ]trapping\b",
                r"\bmosaic (attenuation|perfusion|pattern)\b",
                r"\bhyperinflation\b",
                r"\bbronchiect\w*\b",
            ),
            notes=(
                "Asthma is a clinical/PFT diagnosis; CT concepts are associated airway "
                "findings (bronchial wall thickening, air-trapping, mosaic attenuation, "
                "hyperinflation) only."
            ),
            source="auto_label",
        )
    if "breast" in l and ("cancer" in l or "neoplasm" in l or "carcinoma" in l or "malignan" in l):
        return make_profile(
            label,
            direct=(
                r"\bbreast (mass|carcinoma|malignan\w*)\b",
                r"\b(spiculated|irregular|suspicious)\b.{0,20}\bbreast\b",
                r"\bbreast\b.{0,20}\b(spiculated|irregular|suspicious|mass)\b",
            ),
            associated=(
                r"\baxillary (lymph\w*|node\w*|adenopathy)\b",
                r"\bbreast nodule\b",
            ),
            exclude=(
                r"\bcyst\b",
                r"\bbenign\b",
                r"\bimplant\b",
            ),
            notes=(
                "Breast malignancy: a suspicious breast mass and axillary adenopathy "
                "are visible when the breast is within the chest-CT field of view; "
                "benign cysts/implants excluded. FOV-dependent — review for CTPA "
                "coverage."
            ),
            source="auto_label",
        )
    if "heart valve" in l or "valvular" in l or ("valve" in l and "rheumatic" in l):
        return make_profile(
            label,
            associated=(
                r"\bvalvular calcif\w*\b",
                r"\baortic valve calcif\w*\b",
                r"\bmitral (valve |annular )?calcif\w*\b",
                r"\bvalve (thickening|calcif\w*)\b",
            ),
            notes=(
                "Valvular heart disease: CT concepts are valvular/annular calcification "
                "associations; valve function is not directly assessed on CT."
            ),
            source="auto_label",
        )
    if "lymphadenitis" in l:
        return make_profile(
            label,
            associated=(
                r"\blymphadenopathy\b",
                r"\benlarged lymph nodes?\b",
                r"\breactive lymph nodes?\b",
                r"\blymph node\w*\b.{0,20}\b(enlarg\w*|prominent|reactive)\b",
            ),
            notes=(
                "Lymphadenitis: enlarged/reactive lymph nodes are the CT proxy; "
                "retained as associated (non-specific) findings."
            ),
            source="auto_label",
        )

    return make_profile(
        label,
        notes="No defensible automatic CT concept profile; keep out of trusted concept-space modeling until reviewed.",
        source="review_empty",
    )


def compile_all(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [rx(p) for p in patterns]


def slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s or "label"


def matched_patterns(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    return [p.pattern for p in patterns if p.search(text)]


def load_concepts(path: Path) -> list[str]:
    arr = np.load(path, allow_pickle=True)["concepts"]
    return [str(x) for x in arr]


def load_audit(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_annotation_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        annotations = json.load(f)
    inspect = annotations.get("inspect", {})
    return {
        label: entry.get("category", "")
        for label, entry in inspect.items()
        if isinstance(entry, dict)
    }


def profile_signature(profile: LabelProfile) -> tuple:
    return (profile.direct, profile.associated, profile.exclude)


def classify_concept(
    concept: str,
    profile: LabelProfile,
    global_exclude: list[re.Pattern[str]],
    direct_patterns: list[re.Pattern[str]],
    assoc_patterns: list[re.Pattern[str]],
    label_exclude: list[re.Pattern[str]],
) -> tuple[str, int, list[str], list[str]]:
    global_hits = matched_patterns(concept, global_exclude)
    label_excl_hits = matched_patterns(concept, label_exclude)
    if global_hits or label_excl_hits:
        return "excluded", 0, [], global_hits + label_excl_hits

    direct_hits = matched_patterns(concept, direct_patterns)
    assoc_hits = matched_patterns(concept, assoc_patterns)
    if direct_hits:
        score = 100 + 5 * len(direct_hits) - min(len(concept.split()), 30)
        return "direct", score, direct_hits, []
    if assoc_hits:
        score = 50 + 5 * len(assoc_hits) - min(len(concept.split()), 30)
        return "associated", score, assoc_hits, []
    return "rejected", 0, [], []


def load_top10(audit: dict) -> dict[str, list[str]]:
    return {
        label: [item["concept"] for item in entry.get("positive", [])[:10]]
        for label, entry in audit.items()
    }


def scan_profile(
    concepts: list[str],
    profile: LabelProfile,
    global_exclude: list[re.Pattern[str]],
) -> tuple[list[dict], dict[str, int]]:
    if not profile.direct and not profile.associated:
        return [], {
            "direct": 0,
            "associated": 0,
            "excluded": 0,
            "rejected": len(concepts),
        }

    direct_patterns = compile_all(profile.direct)
    assoc_patterns = compile_all(profile.associated)
    label_exclude = compile_all(profile.exclude)
    trusted_rows: list[dict] = []
    counts = {"direct": 0, "associated": 0, "excluded": 0, "rejected": 0}

    for idx, concept in enumerate(concepts):
        tier, score, hits, _ = classify_concept(
            concept,
            profile,
            global_exclude,
            direct_patterns,
            assoc_patterns,
            label_exclude,
        )
        counts[tier] += 1
        if tier in {"direct", "associated"}:
            trusted_rows.append({
                "tier": tier,
                "score": score,
                "concept_index": idx,
                "concept": concept,
                "matched_patterns": " | ".join(hits),
            })

    trusted_rows.sort(key=lambda r: (-int(r["score"]), r["tier"], r["concept"]))
    return trusted_rows, counts


def profile_status(profile: LabelProfile, counts: dict[str, int]) -> str:
    if not profile.direct and not profile.associated:
        return "review_empty"
    if counts["direct"] > 0 and counts["associated"] > 0:
        return "direct_plus_associated"
    if counts["direct"] > 0:
        return "direct"
    if counts["associated"] > 0:
        return "associated_only"
    return "review_no_matches"


def radiology_flag(row: dict) -> str:
    trusted_total = int(row["trusted_total"])
    direct = int(row["direct_candidates"])
    top10_trusted = int(row["top10_trusted"])
    if trusted_total == 0:
        return "no_trusted_space_review_or_exclude"
    if direct == 0:
        return "associated_only_not_direct_diagnosis"
    if top10_trusted == 0:
        return "trusted_space_exists_but_current_top10_untrusted"
    if top10_trusted < 5:
        return "partial_alignment_review_top10"
    return "usable_with_tier_caveat"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_bank", default=str(ROOT / "concept_bank.openai_emb.npz"))
    ap.add_argument(
        "--audit_json",
        default=str(ROOT / "outputs/v1/audit/phenotype__linear_openai__concept_importance.json"),
    )
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/v1/audit/trusted_concept_spaces"))
    ap.add_argument(
        "--annotations_json",
        default=str(ROOT / "concept_annotations.json"),
        help="Optional concept annotation file used to borrow existing label categories.",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels to process; by default all labels in the audit JSON are processed.",
    )
    ap.add_argument(
        "--max_per_tier",
        type=int,
        default=0,
        help="Maximum rows to write per trusted tier; 0 keeps all matching concepts.",
    )
    args = ap.parse_args()

    concept_path = Path(args.concept_bank)
    out_dir = Path(args.out_dir)
    concepts = load_concepts(concept_path)
    audit = load_audit(Path(args.audit_json))
    categories = load_annotation_categories(Path(args.annotations_json))
    labels = args.labels if args.labels is not None else list(audit)
    top10_by_label = load_top10(audit)
    global_exclude = compile_all(GLOBAL_EXCLUDE)
    scan_cache: dict[tuple, tuple[list[dict], dict[str, int]]] = {}

    summary_rows: list[dict] = []
    manifest: dict[str, object] = {
        "concept_bank": str(concept_path),
        "audit_json": str(Path(args.audit_json)),
        "annotations_json": str(Path(args.annotations_json)),
        "n_concepts": len(concepts),
        "n_labels": len(labels),
        "labels": {},
    }

    for label in labels:
        profile = auto_profile(label, categories.get(label, ""))
        direct_patterns = compile_all(profile.direct)
        assoc_patterns = compile_all(profile.associated)
        label_exclude = compile_all(profile.exclude)
        signature = profile_signature(profile)
        if signature not in scan_cache:
            scan_cache[signature] = scan_profile(concepts, profile, global_exclude)
        trusted_rows, counts = scan_cache[signature]

        direct_rows = [r for r in trusted_rows if r["tier"] == "direct"]
        assoc_rows = [r for r in trusted_rows if r["tier"] == "associated"]
        if args.max_per_tier > 0:
            direct_rows = direct_rows[: args.max_per_tier]
            assoc_rows = assoc_rows[: args.max_per_tier]
        kept_rows = [
            {"label": label, **row}
            for row in direct_rows + assoc_rows
        ]

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
                concept,
                profile,
                global_exclude,
                direct_patterns,
                assoc_patterns,
                label_exclude,
            )
            top10_counts[tier] += 1
            top10_rows.append({
                "label": label,
                "rank": rank,
                "tier": tier,
                "score": score,
                "concept": concept,
                "matched_patterns": " | ".join(hits),
                "exclusion_patterns": " | ".join(excludes),
            })
        write_csv(
            out_dir / f"{label_slug}.current_top10_audit.csv",
            top10_rows,
            ["label", "rank", "tier", "score", "concept", "matched_patterns", "exclusion_patterns"],
        )

        summary = {
            "label": label,
            "auc": audit.get(label, {}).get("stats", {}).get("test_auc"),
            "category": categories.get(label, ""),
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
        "label",
        "auc",
        "category",
        "profile_source",
        "profile_status",
        "radiology_flag",
        "direct_candidates",
        "associated_candidates",
        "trusted_total",
        "trusted_fraction",
        "top10_direct",
        "top10_associated",
        "top10_trusted",
        "top10_excluded",
        "top10_rejected",
        "notes",
        "trusted_csv",
        "top10_audit_csv",
    ]
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        summary_fields,
    )
    write_csv(
        out_dir / "radiology_review_flags.csv",
        summary_rows,
        [
            "label",
            "auc",
            "category",
            "profile_source",
            "profile_status",
            "radiology_flag",
            "direct_candidates",
            "associated_candidates",
            "trusted_total",
            "top10_trusted",
            "notes",
        ],
    )
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"loaded {len(concepts)} concepts")
    print(f"processed {len(labels)} labels using {len(scan_cache)} unique profile scans")
    print(f"wrote {out_dir}")
    for row in summary_rows:
        print(
            f"{row['label']}: trusted={row['trusted_total']} "
            f"direct={row['direct_candidates']} assoc={row['associated_candidates']} "
            f"top10 trusted={row['top10_direct'] + row['top10_associated']}/10"
        )


if __name__ == "__main__":
    main()
