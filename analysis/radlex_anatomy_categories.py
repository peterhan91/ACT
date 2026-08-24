#!/usr/bin/env python3
"""Assign concept-bank phrases to RadLex-anchored anatomical groups.

The groups are project-defined unions anchored to RadLex 4.3; they are not
official RadLex classes.  The classifier is deliberately conservative: it uses
explicit anatomy/finding phrases plus documented clinical tie-breaks, and sends
unresolved, device/technical, or out-of-scope concepts to ``Other``.

This script never rewrites the existing finding labels in ``categories.npy``.
It creates a separate row-aligned vector for anatomical coloring of UMAP panel
A, along with counts and a reproducibility/audit record.

Usage:
    python radlex_anatomy_categories.py
    python radlex_anatomy_categories.py --audit-examples 12
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_UMAP_DIR = ROOT / "outputs" / "concept_umap" / "f2llm"
RADLEX_OWL = ROOT / "data" / "radlex" / "4.3" / "RadLex.owl"
RADLEX_VERSION = "4.3"
RADLEX_SOURCE = "https://radlex.org/download/RadLex_Owl4.3.zip"
OTHER = "Other"

# Figure order follows anatomy rather than prevalence.  Keep this synchronized
# with data/radlex/COARSE_GROUPS_PROPOSAL.md.
GROUPS = (
    "Lung parenchyma",
    "Airways",
    "Pleura",
    "Mediastinum",
    "Lymph nodes/lymphatics",
    "Heart/pericardium",
    "Vasculature",
    "Hepatobiliary",
    "Pancreas/spleen/adrenals",
    "Gastrointestinal tract",
    "Peritoneum/retroperitoneum",
    "Renal/urinary",
    "Reproductive organs",
    "Musculoskeletal/body wall",
    OTHER,
)

# Compact roots from the proposal plus a small set of explicitly reviewed seed
# structures that are not reliably reachable through RadLex's incomplete part
# hierarchy.  The proposal intentionally displays only the compact roots.
RADLEX_ANCHORS = {
    "Lung parenchyma": {
        "RID1301": "lung", "RID13437": "lungs", "RID35739": "lung parenchyma",
    },
    "Airways": {
        "RID1245": "airway", "RID1246": "tracheobronchial tree", "RID1298": "bronchiole",
    },
    "Pleura": {
        "RID1362": "pleura", "RID1363": "pleural space", "RID1370": "parietal pleura",
        "RID1371": "visceral pleura", "RID39203": "pleural sac",
    },
    "Mediastinum": {
        "RID1384": "mediastinum", "RID43249": "subdivision of mediastinum", "RID1430": "thymus",
    },
    "Lymph nodes/lymphatics": {
        "RID13296": "lymph node",
        "RID30383": "lymphatic vessel",
        "RID28847": "lymph node group",
        "RID28922": "lymphatic chain",
    },
    "Heart/pericardium": {
        "RID1385": "heart", "RID1407": "pericardium", "RID1398": "myocardium",
        "RID1399": "endocardium", "RID34906": "cardiac septum",
    },
    "Vasculature": {
        "RID4": "blood vessel",
        "RID13183": "arterial system",
        "RID13184": "venous system",
    },
    "Hepatobiliary": {
        "RID58": "liver",
        "RID28753": "biliary system",
        "RID187": "gallbladder",
        "RID191": "bile duct",
    },
    "Pancreas/spleen/adrenals": {
        "RID170": "pancreas",
        "RID86": "spleen",
        "RID88": "adrenal gland",
    },
    "Gastrointestinal tract": {"RID94": "gastrointestinal tract"},
    "Peritoneum/retroperitoneum": {
        "RID410": "peritoneum",
        "RID397": "peritoneal cavity",
        "RID431": "retroperitoneum",
        "RID33180": "mesentery",
        "RID29251": "omentum",
        "RID439": "extraperitoneal space",
    },
    "Renal/urinary": {
        "RID204": "urinary tract",
        "RID205": "kidney",
        "RID39347": "urinary system",
        "RID223": "renal collecting system",
        "RID229": "ureter",
        "RID237": "urinary bladder",
        "RID34890": "urethra",
    },
    "Reproductive organs": {
        "RID39057": "genital system",
        "RID270": "genital system of female human body",
        "RID342": "genital system of male human body",
    },
    "Musculoskeletal/body wall": {
        "RID13209": "musculoskeletal system",
        "RID30066": "body wall",
        "RID13197": "bone organ",
        "RID13196": "muscle organ",
        "RID6122": "joint",
        "RID7741": "spine",
        "RID38594": "bone marrow",
        "RID1524": "diaphragm",
        "RID2468": "chest wall",
        "RID30014": "wall of abdomen",
        "RID35730": "portion of soft tissue",
    },
    OTHER: {},
}

# Fixed, high-contrast display palette.  Other is intentionally neutral and is
# always drawn first as background.
ANATOMY_COLORS = {
    "Lung parenchyma": "#168B53",
    "Airways": "#E68613",
    "Pleura": "#4C78A8",
    "Mediastinum": "#B279A2",
    "Lymph nodes/lymphatics": "#8C564B",
    "Heart/pericardium": "#C51B2D",
    "Vasculature": "#17A8B5",
    "Hepatobiliary": "#8F9B22",
    "Pancreas/spleen/adrenals": "#7554A3",
    "Gastrointestinal tract": "#D99C16",
    "Peritoneum/retroperitoneum": "#73B839",
    "Renal/urinary": "#F2776B",
    "Reproductive organs": "#D94F9D",
    "Musculoskeletal/body wall": "#536878",
    OTHER: "#D9D9D9",
}


@dataclass(frozen=True)
class Rule:
    name: str
    weight: int
    pattern: re.Pattern[str]


def _rule(name: str, weight: int, pattern: str) -> Rule:
    return Rule(name=name, weight=weight, pattern=re.compile(pattern, re.IGNORECASE))


# Scores encode specificity, not statistical confidence: 6 = an essentially
# group-defining finding, 4 = explicit anatomy, 2 = useful but contextual site.
RULES: dict[str, tuple[Rule, ...]] = {
    "Lung parenchyma": (
        _rule("lung-specific finding", 6, r"\b(?:emphysema|atelecta(?:sis|tic)|pneumoni(?:a|tis)|bronchopneumonia|pulmonary edema|ground[- ]glass(?: opacity)?|airspace (?:opacity|disease)|alveolar (?:opacity|disease)|honeycomb(?:ing)?|interstitial (?:lung disease|thickening|opacity)|pulmonary fibrosis|consolidation|pleuroparenchymal|peribronchovascular)\b"),
        _rule("lung anatomy", 4, r"\b(?:lungs?|pulmonary parenchyma|lung parenchyma|intrapulmonary)\b"),
        _rule("pulmonary qualifier", 3, r"\bpulmonary\b"),
        _rule("lung lobe/site", 3, r"\b(?:(?:right|left|bilateral) )?(?:upper|middle|lower) lobes?\b|\blingul(?:a|ar)\b|\bbibasilar\b"),
        _rule("peripheral lung site", 2, r"\b(?:subpleural|perifissural|intrafissural)\b"),
    ),
    "Airways": (
        _rule("airway-specific finding", 6, r"\b(?:bronchiecta(?:sis|tic)|bronchiolitis|tracheobronchomalacia|mucus plug(?:ging)?|mucoid impaction|endobronchial|airway wall thickening|bronchial wall thickening|tree-in-bud)\b"),
        _rule("airway anatomy", 4, r"\b(?:airways?|bronch(?:us|i|ial|iole|iolar)|trachea|tracheal|tracheobronchial|carina|carinal)\b"),
    ),
    "Pleura": (
        _rule("pleural-space finding", 6, r"\b(?:pleural effusions?|pneumothora(?:x|ces)|hydropneumothora(?:x|ces)|hemothora(?:x|ces)|empyema)\b"),
        _rule("pleural anatomy", 4, r"\b(?:pleura|pleural|extrapleural)\b"),
    ),
    "Mediastinum": (
        _rule("mediastinal finding", 6, r"\b(?:pneumomediastinum|mediastinitis)\b"),
        _rule("thymic anatomy", 5, r"\b(?:thymus|thymic)\b"),
        _rule("mediastinal anatomy", 4, r"\b(?:mediastinum|mediastinal)\b"),
        _rule("mediastinal compartment", 2, r"\b(?:prevascular|subcarinal|paratracheal|aortopulmonary window)\b"),
    ),
    "Lymph nodes/lymphatics": (
        _rule("lymph node", 6, r"\b(?:lymph nodes?|lymphadenopath(?:y|ies)|lymphadenitis|adenopath(?:y|ies)|nodal chain|nodal disease)\b"),
        _rule("node shorthand", 5, r"\bnodes?\b"),
        _rule("lymphatic anatomy", 4, r"\blymphatic(?:s| vessels?)?\b"),
    ),
    "Heart/pericardium": (
        _rule("pericardial/cardiac finding", 6, r"\b(?:cardiomegaly|pericardi\w*|myocardi\w*|cardiac tamponade)\b"),
        _rule("heart anatomy", 4, r"\b(?:heart|cardiac|myocardium|endocardium|atria|atrium|atrial|ventricles?|ventricular|interventricular)\b"),
        _rule("cardiac valve", 5, r"\b(?:(?:aortic|mitral|tricuspid|pulmonic) valves?|valvular)\b"),
    ),
    "Vasculature": (
        _rule("vascular disease", 6, r"\b(?:atherosclero\w*|aneurysm\w*|pseudoaneurysm\w*|dissect(?:ion|ed|ing)|thromboembol\w*|embol(?:us|i|ism|ic)|thrombo(?:sis|sed)|thromb(?:us|i|osis|osed)|phlebitis|varix|varices|(?:arteriovenous|portosystemic|splenorenal|portocaval) (?:malformation|fistula|shunt)s?|arterial[- ]portal shunts?)\b"),
        _rule("vessel anatomy", 5, r"\b(?:aorta|aortic|artery|arteries|vein|veins|vasculature|vascular|vessels?|vena cava|caval|ivc|svc|sma|smv|ima|imv|celiac (?:axis|trunk)|(?:portal|portosplenic|splenoportal|portomesenteric) confluence)\b"),
        _rule("qualified vessel finding", 5, r"\b(?:vascular|arterial|venous) (?:stenosis|occlusion|calcification|congestion)\b"),
    ),
    "Hepatobiliary": (
        _rule("liver/biliary anatomy", 5, r"\b(?:liver|hepatic|hepatobiliary|porta hepatis|gallbladder|biliary|bile ducts?|common bile duct|cystic duct)\b"),
        _rule("liver/biliary finding", 6, r"\b(?:hepatomegaly|steato(?:sis|tic)|cirrho(?:sis|tic)|cholecyst\w*|cholelithiasis|gallstones?|choledocholithiasis|portal hypertension)\b"),
    ),
    "Pancreas/spleen/adrenals": (
        _rule("upper-abdominal organ", 5, r"\b(?:pancrea\w*|spleen|splenic|splenomegaly|splenule|adrenals?|suprarenal)\b"),
        _rule("pancreatic abbreviation", 6, r"\b(?:ipmn|pdac)\b"),
    ),
    "Gastrointestinal tract": (
        _rule("GI anatomy", 5, r"\b(?:gastrointestinal|esophag\w*|stomach|gastric|gastroduodenal|duoden\w*|small bowel|large bowel|bowel|intestin\w*|jejun\w*|ileum|ileal|terminal ileum|colon|colonic|cecum|cecal|sigmoid|rectum|rectal|anorectal|anal canal|appendix|appendiceal|ileostomy|colostomy|gastrostomy)\b"),
        _rule("GI-specific finding", 6, r"\b(?:appendicitis|colitis|enteritis|diverticul\w*|gastritis|esophagitis|proctitis|intersphincteric|fecal burden|faecal burden|colectomy|hartmann(?:'s)? pouch)\b"),
        _rule("GI hernia", 7, r"\b(?:hiatal|internal) hernias?\b"),
    ),
    "Peritoneum/retroperitoneum": (
        _rule("peritoneal anatomy", 5, r"\b(?:peritoneum|peritoneal|intraperitoneal|extraperitoneal|retroperitoneum|retroperitoneal|mesentery|mesenteric|omentum|omental|para-?colic gutter|peri-?colic gutter|presacral space)\b"),
        _rule("peritoneal-space finding", 6, r"\b(?:ascites|pneumoperitoneum|hemoperitoneum|free intraperitoneal (?:air|fluid)|free (?:air|fluid) in the (?:abdomen|pelvis)|pelvic free fluid)\b"),
        _rule("free air/fluid", 4, r"\bfree (?:air|fluid)\b"),
    ),
    "Renal/urinary": (
        _rule("urinary anatomy", 5, r"\b(?:kidneys?|renal|nephric|ureters?|ureteral|ureteric|urinary|urothelial|urinary bladder|bladder|urethra|urethral|collecting system|renal pelvis|calyces|calyceal|interpolar|perinephric)\b"),
        _rule("urinary-specific finding", 6, r"\b(?:hydronephro\w*|hydroureter\w*|nephro(?:lithiasis|calcinosis)|pyelonephritis|nonobstructing (?:stone|calculus|calculi)|renal calcul(?:us|i)|ureteral calcul(?:us|i))\b"),
    ),
    "Reproductive organs": (
        _rule("reproductive anatomy", 5, r"\b(?:genital system|uterus|uterine|endometrium|endometrial|myometrium|ovary|ovaries|ovarian|adnexa|adnexal|fallopian|cervix|vagina|vaginal|vulva|vulvar|prostate|prostatic|seminal vesicles?|testis|testes|testicular|scrotum|scrotal|penis|penile|parametrium|parametrial|mesosalpinx|mesovarium|broad ligament)\b"),
        _rule("reproductive-specific finding", 6, r"\b(?:endometri\w*|fibroids?|leiomyoma\w*|corpus lute\w*|hydrosalpinx)\b"),
    ),
    "Musculoskeletal/body wall": (
        _rule("bone/spine finding", 6, r"\b(?:fractures?|osseous|bony|intraosseous|sclerotic lesions?|lytic lesions?|bone infarcts?|scoliosis|spondyl\w*|anterolisthesis|retrolisthesis|listhesis|laminectomy|osteolytic|osteoblastic|osteopenia|compression deformity|degenerative disc disease|(?:central|spinal|lumbar|cervical|thoracic) canal stenosis)\b"),
        _rule("bone/joint anatomy", 5, r"\b(?:musculoskeletal system|bones?|vertebra\w*|spine|spinal|ribs?|sternum|clavicle|clavicular|scapula|scapular|humerus|humeral|femur|femoral|hip|acetabul\w*|sacrum|sacral|coccyx|coccygeal|ilium|iliac (?:bone|crest|wing)|ischium|ischial|pubic (?:bone|symphysis|ramus|rami)|joints?|marrow|c[1-7]|t(?:[1-9]|1[0-2])|l[1-5]|s[1-5])\b"),
        _rule("muscle/body-wall anatomy", 5, r"\b(?:muscle organ|skeletal muscle|musculature|intramuscular|psoas|paraspinal muscles?|abdominal wall|chest wall|body wall|subcutaneous (?:tissues?|stranding|emphysema)|diaphragm|diaphragmatic)\b"),
        _rule("body-wall finding", 6, r"\b(?:incisional hernias?|ventral(?: wall)? hernias?|inguinal hernias?|umbilical hernias?|femoral hernias?|parastomal hernias?|spigelian hernias?|peri-?umbilical hernias?|para-?umbilical hernias?|supra-?umbilical hernias?|sub-?(?:xiphoid|xyphoid) hernias?|bochdalek hernias?|morgagni hernias?|lumbar hernias?|grynfellt\)? hernias?|petit hernias?|flank hernias?|intercostal hernias?|pelvic floor hernias?|perineal hernias?|fat[- ]containing(?:\s+[\w()/-]+){0,4}\s+hernias?)\b"),
    ),
}

# Priority applies only after score and first-match position tie.  Broad spatial
# compartments intentionally come last; explicit organ/site rules win first.
TIE_PRIORITY = {name: i for i, name in enumerate((
    "Lymph nodes/lymphatics",
    "Heart/pericardium",
    "Vasculature",
    "Airways",
    "Pleura",
    "Renal/urinary",
    "Reproductive organs",
    "Hepatobiliary",
    "Pancreas/spleen/adrenals",
    "Gastrointestinal tract",
    "Peritoneum/retroperitoneum",
    "Lung parenchyma",
    "Musculoskeletal/body wall",
    "Mediastinum",
))}

NODE_RE = re.compile(
    r"\b(?:lymph nodes?|lymphadenopath(?:y|ies)|adenopath(?:y|ies)|nodal(?: chain| disease)?|nodes?)\b",
    re.IGNORECASE,
)
LYMPHATIC_RE = re.compile(r"\blymphatic(?:s| vessels?)?\b", re.IGNORECASE)
SCHMORL_RE = re.compile(
    r"\bschmorl(?:'s)? (?:nodes?|nodules?|depressions?|impressions?)\b",
    re.IGNORECASE,
)
SCHMORL_DIAGNOSIS_RE = re.compile(
    r"\b(?:represent(?:s|ing)|compatible with|consistent with|likely|most likely).{0,80}\bschmorl(?:'s)? (?:nodes?|nodules?|depressions?|impressions?)\b",
    re.IGNORECASE,
)
SUBCUTANEOUS_EMPHYSEMA_RE = re.compile(
    r"\b(?:(?:subcutaneous|soft tissue) emphysema|emphysema around the (?:port|catheter|drain)|emphysema.{0,120}(?:pacemaker|ports?|catheters?|drains?).{0,120}(?:subcutaneous|adipose tissue|muscle planes?))\b",
    re.IGNORECASE,
)
PULMONARY_LIGAMENT_RE = re.compile(r"\bpulmonary ligament\b", re.IGNORECASE)
FALCIFORM_LIGAMENT_RE = re.compile(r"\bfalciform ligament\b", re.IGNORECASE)
PERICARD_RE = re.compile(r"\bpericardi\w*\b", re.IGNORECASE)
PLEURAL_SPACE_RE = re.compile(
    r"\b(?:(?:pleural|subpleural) (?:effusions?|fluid)|pneumothora(?:x|ces)|hydropneumothora(?:x|ces)|hemothora(?:x|ces)|empyema)\b",
    re.IGNORECASE,
)
AIRWAY_DISEASE_RE = re.compile(
    r"\b(?:bronchiecta(?:sis|tic)|bronchiolitis|tracheobronchomalacia|mucus plug(?:ging)?|endobronchial|airway wall thickening|bronchial wall thickening)\b",
    re.IGNORECASE,
)
AORTIC_VALVE_RE = re.compile(r"\baortic valves?\b", re.IGNORECASE)
VASCULAR_EVENT_RE = re.compile(
    r"\b(?:pulmonary (?:artery )?(?:embol\w*|thromb\w*)|portal vein thromb\w*|deep venous thromb\w*|aortic (?:aneurysm|dissection)|coronary (?:artery |arterial )?(?:calcification|atherosclerosis|calcific(?:ied)? plaques?|plaques?|stenosis))\b",
    re.IGNORECASE,
)
SECONDARY_CUE_RE = re.compile(
    r"\b(?:adjacent to|abutting|near(?![-\s]+(?:complete(?:ly)?|circumferential|resolution|total|occlus(?:ion|ive)|water|fluid|fusion|anatomic|normal|entire|obliteration|obstruction)\b)|anterior to|posterior to|superior to|inferior to|extending (?:to|into|toward|towards)|invad(?:e|es|ed|ing)|involv(?:e|es|ed|ing)|encas(?:e|es|ed|ing)|displac(?:es|ing)|compress(?:es|ed|ing)?|mass effect on|surrounding|subjacent to|adher(?:ed|ent) to)\b",
    re.IGNORECASE,
)
STRICT_LANDMARK_CUE_RE = re.compile(
    r"^(?:adjacent to|abutting|near|anterior to|posterior to|superior to|inferior to|extending (?:to|into|toward|towards)|invad(?:e|es|ed|ing)|encas(?:e|es|ed|ing)|displac(?:es|ing)|compress(?:es|ed|ing)?|mass effect on|subjacent to)$",
    re.IGNORECASE,
)
VP_SHUNT_RE = re.compile(r"\b(?:ventriculoperitoneal|vp) shunts?\b", re.IGNORECASE)
ANEURYSMAL_BONE_CYST_RE = re.compile(r"\baneurysmal bone cyst\b", re.IGNORECASE)
BLADDER_DIVERTIC_RE = re.compile(
    r"\b(?:(?:urinary bladder|bladder|(?:vesico)?urachal).{0,160}diverticul(?:um|a|ar)\w*|diverticul(?:um|a|ar)\w*.{0,160}(?:urinary bladder|bladder|(?:vesico)?urachal))\b",
    re.IGNORECASE,
)
BLADDER_DIVERTIC_DIAGNOSIS_RE = re.compile(
    r"\b(?:possibly|probably|likely|represent(?:s|ing)|compatible with|consistent with).{0,80}\b(?:a )?(?:urinary )?bladder diverticul\w*\b",
    re.IGNORECASE,
)
HEPATIC_FLEXURE_RE = re.compile(r"\bhepatic flexure\b", re.IGNORECASE)
SPLENIC_FLEXURE_RE = re.compile(r"\bsplenic flexure\b", re.IGNORECASE)
PROSTATIC_URETHRA_RE = re.compile(r"\b(?:prostatic urethra|urethral sphincter)\b", re.IGNORECASE)
BILE_DUCT_RE = re.compile(r"\b(?:bile duct|common bile duct|common hepatic duct|cystic duct)\b", re.IGNORECASE)
PANCREATIC_DUCT_RE = re.compile(r"\bpancreatic duct\b", re.IGNORECASE)
AORTIC_STENOSIS_RE = re.compile(r"\baortic stenosis\b", re.IGNORECASE)
SMA_SYNDROME_RE = re.compile(
    r"\b(?:superior mesenteric artery|sma) syndrome\b",
    re.IGNORECASE,
)
NAMED_VESSEL_RE = re.compile(
    r"\b(?:aorta|sma|smv|ima|imv|celiac (?:axis|trunk)|(?:portal|portosplenic|splenoportal|portomesenteric) confluence|(?:[a-z][a-z-]*\s+){0,3}(?:artery|arteries|vein|veins|vena cava|vasculature|vascular structures?|vessels?))\b",
    re.IGNORECASE,
)
DEVICE_RE = re.compile(
    r"\b(?:catheters?|endotracheal tube|enteric tube|feeding tube|nasogastric tube|chest tube|drains?|central lines?|picc(?: line)?|ports?|pacemakers?|defibrillators?|stents?|prosthe(?:sis|ses|tic)|valve replacement|assist devices?|(?:breast|dental|cochlear|orthopedic|prosthetic) implants?|intrauterine devices?|hardware|surgical clips?|embolization coils?|ivc filters?|stimulator devices?)\b",
    re.IGNORECASE,
)
NONDEVICE_FINDING_RE = re.compile(
    r"\b(?:masses?|nodules?|lesions?|effusions?|collections?|abscess(?:es)?|edema|hemorrhage|hematomas?|infection|inflammation|thromb\w*|stenosis|occlusion|fractures?|diverticul\w*|hernias?|hydronephro\w*|dilat\w*|thickening|calcifications?|plaques?|atheroma\w*|atherosclero\w*|atelecta\w*|pneumothora(?:x|ces)|consolidation|opacit\w*|emphysema|fibrosis|metasta\w*|tumou?rs?|cancers?)\b",
    re.IGNORECASE,
)
MULTISYSTEM_CONNECTOR_RE = re.compile(r",|\b(?:and|or)\b", re.IGNORECASE)

SENTINEL_PATTERNS = {
    "pericardial_effusion": r"\bpericardial effusion\b",
    "cardiophrenic_node": r"(?:cardiophrenic.{0,40}\b(?:lymph )?node|\b(?:lymph )?node.{0,40}cardiophrenic)",
    "vp_shunt": r"\b(?:ventriculoperitoneal|vp) shunts?\b",
    "arterially_enhancing": r"\barterially enhancing\b",
    "femoral_neck": r"\bfemoral neck\b",
    "porta_hepatis": r"\bporta hepatis\b",
    "mesenteric_vessel": r"\bmesenteric (?:artery|arteries|vein|veins|vessels?)\b",
    "mesenteric_node": r"\bmesenteric (?:lymph )?nodes?\b",
    "retroperitoneal_node": r"\bretroperitoneal (?:lymph )?nodes?\b",
    "iliac_vessel": r"\biliac (?:artery|arteries|vein|veins)\b",
    "iliac_node": r"\biliac (?:lymph )?nodes?\b",
    "iliac_bone": r"\biliac (?:bone|crest|wing)\b",
    "free_fluid": r"\bfree fluid\b",
    "free_air": r"\bfree air\b",
    "perinephric": r"\bperinephric\b",
    "bladder_diverticulum": r"\bbladder diverticul(?:um|a|ar)\w*\b",
    "hepatic_flexure": r"\bhepatic flexure\b",
    "splenic_flexure": r"\bsplenic flexure\b",
    "prostatic_urethra": r"\bprostatic urethra\b",
    "pancreatic_duct": r"\bpancreatic duct\b",
    "bile_duct": r"\bbile duct\b",
    "mediastinal_vascular_structure": r"\bmediastinal (?:major |main )?vascular structures?\b",
    "schmorl_node": r"\bschmorl(?:'s)? (?:nodes?|nodules?|depressions?|impressions?)\b",
    "subcutaneous_emphysema": r"\bsubcutaneous emphysema\b",
    "displaced_fracture": r"\b(?:displaced.{0,50}fractures?|fractures?.{0,50}displaced)\b",
    "fat_containing_hernia": r"\bfat[- ]containing.{0,60}hernias?\b",
    "bladder_and_diverticulum": r"\b(?:(?:urinary bladder|bladder).{0,80}diverticul(?:um|a|ar)\w*|diverticul(?:um|a|ar)\w*.{0,80}(?:urinary bladder|bladder))\b",
    "arterial_phase_or_enhancement": r"\barterial(?:ly)? (?:phase|enhanc\w*|focus|lesion)\b",
    "portal_venous_phase": r"\bportal venous phase\b",
}

REGRESSION_CASES = {
    "no pulmonary embolism": "Vasculature",
    "moderate pericardial effusion": "Heart/pericardium",
    "small right pleural effusion": "Pleura",
    "enlarged mediastinal lymph node": "Lymph nodes/lymphatics",
    "dilated pelvic lymphatic vessel": "Lymph nodes/lymphatics",
    "aortic valve calcification": "Heart/pericardium",
    "portal vein thrombosis": "Vasculature",
    "hepatic vasculature appears patent": "Vasculature",
    "mesenteric mass encasing the mesenteric vessels": "Peritoneum/retroperitoneum",
    "hepatic lesion adjacent to the portal vein": "Hepatobiliary",
    "right renal lesion in the retroperitoneum": "Renal/urinary",
    "multiple bladder diverticula": "Renal/urinary",
    "wall thickening at the hepatic flexure": "Gastrointestinal tract",
    "dilated pancreatic duct": "Pancreas/spleen/adrenals",
    "dilated common bile duct at the pancreatic head": "Hepatobiliary",
    "fracture of the femoral neck": "Musculoskeletal/body wall",
    "ventriculoperitoneal shunt catheter in place": OTHER,
    "subpleural nodule in the right lower lobe": "Lung parenchyma",
    "mild cylindrical bronchiectasis": "Airways",
    "right pleural drain present": OTHER,
    "left thyroid nodule extending towards the anterior mediastinum": OTHER,
    "associated schmorl's node": "Musculoskeletal/body wall",
    "subcutaneous emphysema in the right abdominal wall": "Musculoskeletal/body wall",
    "emphysema around the port": "Musculoskeletal/body wall",
    "soft tissue emphysema secondary to drain placement": "Musculoskeletal/body wall",
    "emphysema observed between the pacemaker and subcutaneous adipose tissue and muscle planes": "Musculoskeletal/body wall",
    "mildly displaced fracture of the left posterior rib": "Musculoskeletal/body wall",
    "fat-containing ventral hernia": "Musculoskeletal/body wall",
    "fat-containing periumbilical hernia": "Musculoskeletal/body wall",
    "fat-containing internal hernia": "Gastrointestinal tract",
    "small fat-containing right upper quadrant ventral wall hernia": "Musculoskeletal/body wall",
    "fat containing sub-xyphoid hernia": "Musculoskeletal/body wall",
    "fat-containing lesion in the right kidney": "Renal/urinary",
    "small diverticulum along the anterior urinary bladder": "Renal/urinary",
    "vesicourachal diverticulum present in the bladder": "Renal/urinary",
    "cystic lesion adjacent to the right posterior bladder, possibly a bladder diverticulum": "Renal/urinary",
    "arterially enhancing focus in the right hepatic lobe": "Hepatobiliary",
    "no arterial phase performed": OTHER,
    "portal venous phase": OTHER,
    "near complete destruction of the left femoral neck": "Musculoskeletal/body wall",
    "near complete atelectasis of the left lower lobe": "Lung parenchyma",
    "near-complete atelectasis of the left lower lobe": "Lung parenchyma",
    "near-complete disc height loss at l5-s1": "Musculoskeletal/body wall",
    "near-occlusive thrombus in the femoral vein": "Vasculature",
    "near-resolution of left hydronephrosis": "Renal/urinary",
    "bilateral moderate hydronephrosis despite bilateral ureteral stents": "Renal/urinary",
    "biliary duct dilatation with concern for a blocked stent": "Hepatobiliary",
    "infiltrative thickening of the gastric antrum with a stent": "Gastrointestinal tract",
    "severe coronary artery calcification and stents": "Vasculature",
    "severe coronary calcific plaque with stents": "Vasculature",
    "left pneumothorax with a pigtail pleural drain": "Pleura",
    "central canal stenosis": "Musculoskeletal/body wall",
    "pulmonary ligament thickening": "Pleura",
    "falciform ligament thickening": "Peritoneum/retroperitoneum",
    "parametrial soft tissue mass": "Reproductive organs",
    "alveolar opacity in the lower lobes": "Lung parenchyma",
    "interstitial thickening in both lungs": "Lung parenchyma",
    "peritoneal soft tissue implant": "Peritoneum/retroperitoneum",
    "splenorenal shunt with gastric varices": "Vasculature",
    "near-complete collapse of smv by mass": "Vasculature",
    "50 percent narrowing of the proximal sma": "Vasculature",
    "superior mesenteric artery syndrome": "Gastrointestinal tract",
}


@dataclass(frozen=True)
class Assignment:
    group: str
    reason: str
    score: int
    tied: bool
    candidates: tuple[str, ...]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("–", "-").replace("—", "-").split())


def classify(text: str, *, _ignore_secondary: bool = False) -> Assignment:
    """Assign one phrase and return both the group and auditable rationale."""
    s = _normalize(text)
    split = [s] if _ignore_secondary else SECONDARY_CUE_RE.split(s, maxsplit=1)
    scope = split[0]

    # Clinically explicit tie-breaks from the group specification.
    if VP_SHUNT_RE.search(scope):
        return Assignment(OTHER, "override:device-vp-shunt", 9, False, (OTHER,))
    if SCHMORL_RE.search(scope) or SCHMORL_DIAGNOSIS_RE.search(s):
        return Assignment("Musculoskeletal/body wall", "override:schmorl-node", 9, False, ("Musculoskeletal/body wall",))
    if SUBCUTANEOUS_EMPHYSEMA_RE.search(scope):
        return Assignment("Musculoskeletal/body wall", "override:subcutaneous-emphysema", 9, False, ("Musculoskeletal/body wall",))
    if NODE_RE.search(scope) and not SCHMORL_RE.search(scope):
        return Assignment("Lymph nodes/lymphatics", "override:lymph-node", 9, False, ("Lymph nodes/lymphatics",))
    if LYMPHATIC_RE.search(scope):
        return Assignment("Lymph nodes/lymphatics", "override:lymphatic", 9, False, ("Lymph nodes/lymphatics",))
    if DEVICE_RE.search(scope) and not NONDEVICE_FINDING_RE.search(scope):
        return Assignment(OTHER, "override:device-technical", 9, False, (OTHER,))
    if (
        BLADDER_DIVERTIC_RE.search(scope)
        or BLADDER_DIVERTIC_DIAGNOSIS_RE.search(s)
        or PROSTATIC_URETHRA_RE.search(scope)
    ):
        return Assignment("Renal/urinary", "override:urinary-compound", 9, False, ("Renal/urinary",))
    if HEPATIC_FLEXURE_RE.search(scope) or SPLENIC_FLEXURE_RE.search(scope):
        return Assignment("Gastrointestinal tract", "override:colonic-flexure", 9, False, ("Gastrointestinal tract",))
    if BILE_DUCT_RE.search(scope):
        return Assignment("Hepatobiliary", "override:bile-duct", 9, False, ("Hepatobiliary",))
    if PANCREATIC_DUCT_RE.search(scope):
        return Assignment("Pancreas/spleen/adrenals", "override:pancreatic-duct", 9, False, ("Pancreas/spleen/adrenals",))
    if ANEURYSMAL_BONE_CYST_RE.search(scope):
        return Assignment("Musculoskeletal/body wall", "override:aneurysmal-bone-cyst", 9, False, ("Musculoskeletal/body wall",))
    if PULMONARY_LIGAMENT_RE.search(scope):
        return Assignment("Pleura", "override:pulmonary-ligament", 9, False, ("Pleura",))
    if FALCIFORM_LIGAMENT_RE.search(scope):
        return Assignment("Peritoneum/retroperitoneum", "override:falciform-ligament", 9, False, ("Peritoneum/retroperitoneum",))
    if AORTIC_VALVE_RE.search(scope) or AORTIC_STENOSIS_RE.search(scope):
        return Assignment("Heart/pericardium", "override:aortic-valve", 9, False, ("Heart/pericardium",))
    if SMA_SYNDROME_RE.search(scope):
        return Assignment("Gastrointestinal tract", "override:sma-syndrome", 9, False, ("Gastrointestinal tract",))
    if PERICARD_RE.search(scope) and PLEURAL_SPACE_RE.search(scope):
        if re.search(r"\b(?:cardiomegaly|heart)\b", scope, re.IGNORECASE):
            return Assignment("Heart/pericardium", "override:primary-cardiac", 9, False, ("Heart/pericardium",))
        return Assignment(OTHER, "multisystem:pleural-pericardial", 9, False, ("Pleura", "Heart/pericardium"))
    if PLEURAL_SPACE_RE.search(scope):
        return Assignment("Pleura", "override:pleural-space", 9, False, ("Pleura",))
    if NAMED_VESSEL_RE.search(scope):
        return Assignment("Vasculature", "override:named-vessel", 9, False, ("Vasculature",))
    if PERICARD_RE.search(scope):
        return Assignment("Heart/pericardium", "override:pericardial", 9, False, ("Heart/pericardium",))
    if AIRWAY_DISEASE_RE.search(scope):
        return Assignment("Airways", "override:airway-disease", 9, False, ("Airways",))
    if VASCULAR_EVENT_RE.search(scope):
        return Assignment("Vasculature", "override:vascular-event", 9, False, ("Vasculature",))

    scored: list[tuple[int, int, int, str, tuple[str, ...]]] = []
    for group, rules in RULES.items():
        hits: list[str] = []
        score = 0
        first = len(s) + 1
        for rule in rules:
            match = rule.pattern.search(scope)
            if match:
                hits.append(rule.name)
                score += rule.weight
                first = min(first, match.start())
        if score:
            scored.append((score, -first, -TIE_PRIORITY[group], group, tuple(hits)))

    if not scored:
        if len(split) > 1:
            cue = SECONDARY_CUE_RE.search(s)
            if cue is not None and not STRICT_LANDMARK_CUE_RE.fullmatch(cue.group(0)):
                fallback = classify(s, _ignore_secondary=True)
                if fallback.group != OTHER:
                    return Assignment(
                        fallback.group,
                        "full-scope-fallback:" + fallback.reason,
                        fallback.score,
                        fallback.tied,
                        fallback.candidates,
                    )
            return Assignment(OTHER, "secondary-site-only", 0, False, (OTHER,))
        return Assignment(OTHER, "no-explicit-site", 0, False, (OTHER,))

    scored.sort(reverse=True)
    top_score = scored[0][0]
    top_groups = tuple(row[3] for row in scored if row[0] == top_score)
    if len(top_groups) > 1 and MULTISYSTEM_CONNECTOR_RE.search(scope):
        return Assignment(OTHER, "multisystem-equal-specificity", top_score, True, top_groups)
    winner = scored[0]
    return Assignment(
        group=winner[3],
        reason="rules:" + "+".join(winner[4]),
        score=winner[0],
        tied=len(top_groups) > 1,
        candidates=tuple(row[3] for row in scored),
    )


def _fingerprint_strings(strings: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in strings:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_specification() -> int:
    if tuple(ANATOMY_COLORS) != GROUPS:
        raise AssertionError("ANATOMY_COLORS must exactly follow GROUPS order")
    failures = []
    for phrase, expected in REGRESSION_CASES.items():
        actual = classify(phrase).group
        if actual != expected:
            failures.append(f"{phrase!r}: expected {expected!r}, got {actual!r}")
    if failures:
        raise AssertionError("rule regression failures:\n" + "\n".join(failures))

    owl = RADLEX_OWL.read_bytes()
    anchor_rids = {rid for anchors in RADLEX_ANCHORS.values() for rid in anchors}
    missing = [rid for rid in sorted(anchor_rids) if f'/RID/{rid}"'.encode() not in owl]
    if missing:
        raise AssertionError(f"RadLex anchors absent from local OWL: {missing}")
    return len(anchor_rids)


def _spaced_examples(indices: list[int], concepts: np.ndarray, n: int) -> list[str]:
    if not indices or n <= 0:
        return []
    positions = np.linspace(0, len(indices) - 1, min(n, len(indices)), dtype=int)
    return [str(concepts[indices[int(pos)]]) for pos in positions]


def build(bank: Path, umap_dir: Path, audit_examples: int) -> None:
    validated_anchor_count = _validate_specification()
    with np.load(bank, allow_pickle=True) as z:
        concepts = z["concepts"]
    xy_path = umap_dir / "umap2d.npy"
    finding_path = umap_dir / "categories.npy"
    xy = np.load(xy_path, mmap_mode="r")
    finding = np.load(finding_path, allow_pickle=True)
    if not (len(concepts) == len(xy) == len(finding)):
        raise ValueError(
            f"row mismatch: concepts={len(concepts)}, UMAP={len(xy)}, finding labels={len(finding)}"
        )

    labels: list[str] = []
    reason_labels: list[str] = []
    assignment_scores: list[int] = []
    assignment_ties: list[bool] = []
    candidate_counts: list[int] = []
    reasons: Counter[str] = Counter()
    score_counts: Counter[int] = Counter()
    conflict_counts: Counter[str] = Counter()
    conflict_indices: dict[str, list[int]] = defaultdict(list)
    tied_indices: list[int] = []
    group_indices: dict[str, list[int]] = defaultdict(list)

    for i, concept in enumerate(concepts):
        assignment = classify(str(concept))
        labels.append(assignment.group)
        reason_labels.append(assignment.reason)
        assignment_scores.append(assignment.score)
        assignment_ties.append(assignment.tied)
        candidate_counts.append(len(assignment.candidates))
        reasons[assignment.reason] += 1
        score_counts[assignment.score] += 1
        group_indices[assignment.group].append(i)
        if assignment.tied:
            tied_indices.append(i)
        if len(assignment.candidates) > 1:
            key = " | ".join(assignment.candidates[:3])
            conflict_counts[key] += 1
            conflict_indices[key].append(i)

    # Fixed-width ASCII keeps the portable, non-pickled NPY compact; plotting
    # scripts decode it with ``astype(str)``.
    labels_array = np.asarray(labels, dtype=f"S{max(map(len, GROUPS))}")
    unknown = set(np.unique(labels_array).astype(str)) - set(GROUPS)
    if unknown:
        raise AssertionError(f"unexpected anatomy labels: {sorted(unknown)}")

    out_npy = umap_dir / "radlex_anatomy_categories.npy"
    out_audit_npz = umap_dir / "radlex_anatomy_assignment_audit.npz"
    out_json = umap_dir / "radlex_anatomy_categories.metadata.json"
    out_csv = umap_dir / "radlex_anatomy_counts.csv"
    np.save(out_npy, labels_array)
    reason_names = sorted(set(reason_labels))
    reason_to_code = {name: code for code, name in enumerate(reason_names)}
    np.savez_compressed(
        out_audit_npz,
        reason_code=np.fromiter((reason_to_code[x] for x in reason_labels), dtype=np.uint16),
        score=np.asarray(assignment_scores, dtype=np.uint8),
        tied=np.asarray(assignment_ties, dtype=bool),
        candidate_count=np.asarray(candidate_counts, dtype=np.uint8),
    )

    counts = Counter(labels)
    n_total = len(labels)
    sentinel_audits = {}
    for name, pattern in SENTINEL_PATTERNS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        mask = np.fromiter((bool(rx.search(str(x))) for x in concepts), dtype=bool, count=n_total)
        cross_tab = Counter(labels_array[mask].astype(str).tolist())
        sentinel_audits[name] = {
            "pattern": pattern,
            "n": int(mask.sum()),
            "label_counts": {group: cross_tab[group] for group in GROUPS if cross_tab[group]},
        }
    metadata = {
        "schema_version": 1,
        "classifier": Path(__file__).name,
        "radlex": {
            "version": RADLEX_VERSION,
            "source": RADLEX_SOURCE,
            "owl_path": str(RADLEX_OWL.relative_to(ROOT)),
            "owl_sha256": _sha256(RADLEX_OWL),
            "anchors": RADLEX_ANCHORS,
            "validated_anchor_rid_count": validated_anchor_count,
            "qualification": "Project-defined RadLex-anchored anatomical groups; not official RadLex classes.",
        },
        "inputs": {
            "concept_bank": str(bank.relative_to(ROOT) if bank.is_relative_to(ROOT) else bank),
            "concept_count": n_total,
            "concept_order_sha256": _fingerprint_strings(concepts),
            "umap_path": str(xy_path.relative_to(ROOT)),
            "umap_sha256": _sha256(xy_path),
            "finding_categories_path": str(finding_path.relative_to(ROOT)),
            "finding_categories_sha256": _sha256(finding_path),
        },
        "ordered_groups": list(GROUPS),
        "colors": ANATOMY_COLORS,
        "outputs": {
            "categories": str(out_npy.relative_to(ROOT)),
            "categories_sha256": _sha256(out_npy),
            "assignment_audit": str(out_audit_npz.relative_to(ROOT)),
            "assignment_audit_sha256": _sha256(out_audit_npz),
            "reason_code_map": {str(code): name for name, code in reason_to_code.items()},
        },
        "counts": {
            group: {"n": counts[group], "percent": round(100 * counts[group] / n_total, 4)}
            for group in GROUPS
        },
        "audit": {
            "exact_top_score_ties": len(tied_indices),
            "multi_group_rule_matches": sum(conflict_counts.values()),
            "top_candidate_combinations": dict(conflict_counts.most_common(30)),
            "top_candidate_examples": {
                key: _spaced_examples(conflict_indices[key], concepts, min(audit_examples, 12))
                for key, _ in conflict_counts.most_common(15)
            },
            "assignment_reasons": dict(reasons.most_common()),
            "score_distribution": {str(k): v for k, v in sorted(score_counts.items())},
            "examples_by_group": {
                group: _spaced_examples(group_indices[group], concepts, audit_examples)
                for group in GROUPS
            },
            "tie_examples": _spaced_examples(tied_indices, concepts, audit_examples),
            "sentinels": sentinel_audits,
        },
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n")
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n", "percent"])
        for group in GROUPS:
            writer.writerow([group, counts[group], f"{100 * counts[group] / n_total:.4f}"])

    print(f"wrote {out_npy} ({len(labels_array):,} aligned labels)")
    print(f"wrote {out_audit_npz}")
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    for group in GROUPS:
        print(f"  {group:34s} {counts[group]:8,d}  {100 * counts[group] / n_total:6.2f}%")
    print(f"  exact top-score ties: {len(tied_indices):,}")
    print(f"  multi-group rule matches: {sum(conflict_counts.values()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--umap-dir", type=Path, default=DEFAULT_UMAP_DIR)
    parser.add_argument("--audit-examples", type=int, default=12)
    args = parser.parse_args()
    build(args.bank.resolve(), args.umap_dir.resolve(), args.audit_examples)


if __name__ == "__main__":
    main()
