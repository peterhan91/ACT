"""Anatomical / organ-system categorization for the 376,194 radiological concepts.

Each free-text concept (e.g. "small cystic lesion in the pancreatic body
suggestive of a sidebranch ipmn") is assigned to a single best-matching organ-system category by
a prioritized keyword lexicon, mirroring the tissue-category breakdown of
Extended Data Fig. 1 of Lu et al. (Nat Med 2024).

Assignment rule: score each category by the number of distinct keyword hits in
the concept; assign to the highest-scoring category; ties broken by the order in
CATEGORIES (more specific organs listed first). Concepts with no anatomical
keyword fall into "Other / unspecified".
"""
import re

# Ordered most-specific -> least-specific. The KEY is the display name; the value
# is a list of regex fragments matched on word boundaries (case-insensitive).
CATEGORIES = [
    ("Lung & airways", [
        r"lung", r"pulmonary", r"pulmon", r"\blobe", r"bronch\w*", r"alveol\w*",
        r"emphysema", r"bronchiectasis", r"trachea\w*", r"airway", r"parenchym\w*",
        r"ground[- ]?glass", r"interstitial", r"atelecta\w*", r"pneumon\w*",
        r"consolidation", r"reticular", r"honeycomb\w*", r"fibrosis", r"nodule",
        r"opacit\w*", r"infiltrat\w*", r"aspiration",
    ]),
    ("Pleura", [
        r"pleura\w*", r"pleural effusion", r"pneumothorax", r"hydropneumothorax",
        r"effusion",
    ]),
    ("Heart & pericardium", [
        r"heart", r"cardiac", r"cardio\w*", r"cardiomegaly", r"pericard\w*",
        r"coronary", r"ventric\w*", r"atri\w*", r"myocard\w*", r"valv\w*",
        r"mitral", r"tricuspid",
    ]),
    ("Aorta & great vessels", [
        r"aorta", r"aortic", r"vascul\w*", r"arter\w*", r"\bvein\b", r"venous",
        r"\bvena\b", r"\bivc\b", r"\bsvc\b", r"vessel\w*", r"embol\w*",
        r"aneurysm\w*", r"atherosclero\w*", r"plaque", r"thromb\w*", r"caval",
        r"mediastin\w*",
    ]),
    ("Liver & biliary tract", [
        r"liver", r"hepat\w*", r"biliary", r"bile duct\w*", r"intrahepatic",
        r"portal", r"cirrho\w*", r"steatos\w*",
    ]),
    ("Gallbladder", [
        r"gallbladder", r"cholecyst\w*", r"gallstone\w*", r"cholelithiasis",
    ]),
    ("Pancreas", [
        r"pancrea\w*", r"\bipmn\b", r"\bpdac\b",
    ]),
    ("Spleen", [
        r"spleen", r"splen\w*",
    ]),
    ("Adrenal glands", [
        r"adrenal\w*",
    ]),
    ("Kidney & urinary tract", [
        r"kidney\w*", r"renal", r"nephro\w*", r"ureter\w*", r"hydronephro\w*",
        r"bladder", r"urin\w*", r"calcul\w*", r"\bstone\b", r"perinephric",
        r"pyelo\w*",
    ]),
    ("GI tract", [
        r"bowel", r"colon\w*", r"intestin\w*", r"stomach", r"gastric", r"duoden\w*",
        r"rectum", r"rectal", r"sigmoid", r"appendix", r"appendic\w*",
        r"diverticul\w*", r"esophag\w*", r"jejun\w*", r"ileum", r"ileal",
        r"cecum", r"cecal", r"enteric", r"feculent", r"ostomy",
    ]),
    ("Lymph nodes", [
        r"lymph node\w*", r"lymphadenopath\w*", r"\bnode\b", r"\bnodes\b",
        r"adenopath\w*", r"lymphaden\w*",
    ]),
    ("Bone & spine", [
        r"\bbone\w*", r"vertebra\w*", r"\bspine\b", r"spinal", r"osseous",
        r"fracture\w*", r"scoliosis", r"\bdisc\b", r"\brib\b", r"\bribs\b",
        r"sternum", r"sacr\w*", r"osteo\w*", r"degenerative", r"spondyl\w*",
        r"vertebr\w*", r"lumbar", r"thoracic spine", r"pelvic bone", r"\bilium\b",
        r"\bsacrum\b", r"compression deformity", r"marrow",
    ]),
    ("Muscle & soft tissue", [
        r"muscle\w*", r"musculo\w*", r"\bpsoas\b", r"soft tissue", r"subcutaneous",
        r"abdominal wall", r"hernia\w*", r"\bfat\b", r"fatty", r"lipoma\w*",
        r"\bfascia\w*", r"diaphragm\w*",
    ]),
    ("Reproductive & pelvis", [
        r"uter\w*", r"ovar\w*", r"adnex\w*", r"prostat\w*", r"seminal",
        r"endometri\w*", r"cervix", r"cervical cancer", r"vagina\w*", r"fibroid\w*",
        r"\bcyst\b.*ovar", r"corpus luteal", r"testic\w*", r"scrotal",
    ]),
    ("Peritoneum & mesentery", [
        r"ascites", r"peritone\w*", r"free fluid", r"free air", r"mesenter\w*",
        r"omentum", r"omental", r"intraperitoneal", r"retroperiton\w*",
        r"pneumoperitoneum",
    ]),
    ("Thyroid & neck", [
        r"thyroid\w*", r"\bneck\b", r"goiter", r"parathyroid",
    ]),
    ("Breast", [
        r"breast\w*", r"mammar\w*", r"axilla\w*",
    ]),
]

_COMPILED = [
    (name, [re.compile(r"\b" + p if not p.startswith("\\b") and not p.startswith("(") else p,
                        re.IGNORECASE) for p in pats])
    for name, pats in CATEGORIES
]


def categorize(text):
    """Return the best organ-system category name for a concept string."""
    best_name, best_score = "Other / unspecified", 0
    for name, pats in _COMPILED:
        score = sum(1 for rx in pats if rx.search(text))
        if score > best_score:
            best_name, best_score = name, score
    return best_name


if __name__ == "__main__":
    import numpy as np, collections, sys
    z = np.load("concept_bank.f2llm_emb.npz", allow_pickle=True)
    concepts = z["concepts"]
    counts = collections.Counter()
    cat_of = []
    for s in concepts:
        c = categorize(str(s))
        cat_of.append(c)
        counts[c] += 1
    print(f"total = {len(concepts):,}")
    for name, n in counts.most_common():
        print(f"  {name:28s} {n:>8,}  ({100*n/len(concepts):4.1f}%)")
    # save assignment for the plotting script
    np.save("outputs/_concept_category.npy", np.array(cat_of, dtype=object))
    print("saved outputs/_concept_category.npy")
