#!/usr/bin/env python3
r"""
Presence-phrase (regex) dictionaries for phrase-mining CT finding labels.

Two label vocabularies:
  * CHEST_18  — the 18 CT-RATE chest abnormalities (matches LABELS_18 / the
                official CT-RATE RadBERT labeler vocabulary). Used for PMBB chest.
  * ABD_30    — the 30 abdominal-CT findings from Merlin (arXiv 2406.06512,
                Fig 2c; includes 5 lower-thorax findings that abdominal CT sees
                at the lung bases). Used for PMBB abdomen.

Each value is a list of case-insensitive regex patterns whose (un-negated, un-
hedged) match in the FINDINGS/IMPRESSION text => finding PRESENT. Negation and
uncertainty are handled centrally in report_mining.py, so these patterns only
encode the POSITIVE phrasings. Patterns are anatomy-anchored where a bare term
would be ambiguous (e.g. "calcification" -> require aorta/coronary/valve).

The chest set is calibrated against the CT-RATE RadBERT silver labels
(calibrate_chest.py); the abd set is authored from clinical phrasing + spot-
checked (no public silver standard for the Merlin vocabulary).
"""
from __future__ import annotations

# Canonical label names. Order is the y-matrix column order downstream.
CHEST_18 = {
    "Medical material": [
        r"\bmedical material\b", r"\bsurgical material\b",
        r"\bchest tube\b", r"\bthoracostomy tube\b", r"\bpigtail catheter\b",
        r"\bendotracheal tube\b", r"\bet tube\b", r"\btracheostomy\b",
        r"\bnasogastric tube\b", r"\bng tube\b", r"\borogastric tube\b",
        r"\bfeeding tube\b", r"\bdobh?off\b",
        r"\bcentral (?:venous )?(?:catheter|line)\b", r"\bpicc\b",
        r"\bport-?a-?cath\b", r"\b(?:venous|infusion|chest) port\b",
        r"\bpacemaker\b", r"\bpacer\b", r"\bdefibrillator\b", r"\baicd\b",
        r"\bicd lead\b", r"\bpacing lead\b", r"\bcardiac (?:device|lead)\b",
        r"\bsternotomy\b", r"\bsternal wires?\b",
        r"\bsurgical (?:clips?|sutures?|staples?)\b", r"\bsurgical hardware\b",
        r"\bsuture material", r"\bsurgical material", r"\boperation material",
        r"\bmetallic (?:suture|material|densit|fragment|clip)",
        r"\bstent\b", r"\bendovascular (?:stent|graft)\b",
        r"\bprosthe(?:sis|tic)\b", r"\b(?:mechanical |prosthetic )valve\b",
        r"\bsurgical drain\b", r"\bdrainage catheter\b", r"\bswan-?ganz\b",
        r"\bspinal (?:hardware|fusion|rods?)\b",
    ],
    "Arterial wall calcification": [
        r"\barterial (?:wall )?calcific", r"\bvascular calcific",
        r"\bgreat vessel calcific", r"\bcalcified atheromatous", r"\bcalcific atheroma",
        # order-independent: a calcific/atheroma/plaque term near the aorta/great
        # vessels within a clause ("calcified atheromatous plaques ... aortic arch").
        r"\baort\w*\b.{0,80}?\b(?:calcif|atheroma|atherosclerot|plaque)",
        r"\b(?:calcif|atheroma|atherosclerot|plaque)\w*\b.{0,80}?\baort",
    ],
    "Cardiomegaly": [
        r"\bcardiomegaly\b", r"\benlarged (?:cardiac silhouette|heart)\b",
        r"\bcardiac (?:silhouette )?enlarge", r"\bheart (?:is )?enlarged\b",
        r"\benlargement of the (?:cardiac silhouette|heart)\b",
        r"\bcardiac silhouette is enlarged\b",
        # CT-RATE phrasing: "the size of the heart has increased"
        r"\b(?:size of the heart|heart size|cardiac size)\b.{0,15}?\bincreased\b",
        r"\b(?:heart|cardiac silhouette) (?:size )?(?:has |is |have )?increased\b",
        r"\bincreased (?:cardiac silhouette|heart size|cardiothoracic ratio)\b",
    ],
    "Pericardial effusion": [
        r"\bpericardial effusion", r"\bpericardial fluid\b",
        r"\bfluid (?:in|within) the pericard", r"\bpericardial (?:fluid|effusion)\b",
    ],
    "Coronary artery wall calcification": [
        r"\bcoronary (?:artery )?(?:wall )?calcific", r"\bcalcified coronary",
        r"\bcoronary (?:artery )?(?:atheroscleros|disease)\b", r"\bcoronary calcium\b",
        # calcific/atheroma/plaque/stent near "coronary" within a clause.
        r"\bcoronary\b.{0,80}?\b(?:calcif|atheroma|atherosclerot|stent|plaque)",
        r"\b(?:calcif|atheroma|atherosclerot|stent|plaque)\w*\b.{0,80}?\bcoronary",
    ],
    "Hiatal hernia": [
        r"\bhiatal hernia\b", r"\bhiatus hernia\b", r"\bparaesophageal hernia\b",
        r"\bhernia\w* (?:through |of |at )?the (?:esophageal )?hiatus\b",
    ],
    "Lymphadenopathy": [
        r"\blymphadenopathy\b", r"\badenopathy\b",
        r"\benlarged (?:mediastinal |hilar |axillary |paratracheal |subcarinal )?lymph nodes?\b",
        r"\benlarged (?:mediastinal|hilar|axillary|paratracheal) nodes?\b",
        r"\bprominent (?:mediastinal |hilar )?lymph nodes?\b",
        r"\bpathologically enlarged (?:lymph )?nodes?\b",
        r"\bbulky (?:lymph )?nodes?\b", r"\bnodal (?:enlargement|conglomerate)\b",
        # size-based (RadBERT calls >=~1cm short-axis pathologic): lymph node
        # within a clause of a >=10 mm / >=1 cm measurement.
        r"\blymph nodes?\b.{0,70}?\b(?:1[0-9]|[2-9][0-9])\s?mm\b",
        r"\b(?:1[0-9]|[2-9][0-9])\s?mm\b.{0,70}?\blymph nodes?\b",
        r"\blymph nodes?\b.{0,70}?\b1(?:\.\d)?\s?cm\b",
        r"\b(?:reaching|up to|measuring)\b.{0,20}?\b1(?:\.\d)?\s?cm\b.{0,40}?\blymph",
    ],
    "Emphysema": [
        r"\bemphysema\b", r"\bemphysematous\b", r"\bcentrilobular emphysema\b",
        r"\bparaseptal emphysema\b", r"\bbulla\b", r"\bbullae\b", r"\bbullous\b",
    ],
    "Atelectasis": [
        r"\batelecta", r"\bvolume loss\b", r"\blobar collapse\b",
        r"\b(?:lung|lobe|pulmonary) collapse\b", r"\bcollapse of the (?:lung|lobe)\b",
    ],
    "Lung nodule": [
        # bare "nodule" (chest CT context); wrong-organ ("thyroid"/"adrenal"/
        # "hepatic" nodule) and Lung-RADS boilerplate are vetoed via CHEST_EXCLUDE.
        r"\b(?:pulmonary |lung )?nodule", r"\bmicronodule",
        r"\bnodular (?:opacit|densit)", r"\bnoncalcified nodule",
        r"\bsubsolid nodule", r"\bground[- ]glass nodule",
    ],
    "Lung opacity": [
        r"\b(?:lung |pulmonary |airspace |parenchymal )opacit(?:y|ies)\b",
        r"\bopacit(?:y|ies) (?:in|of|within) the (?:lung|right|left|lobe)",
        r"\bground[- ]glass\b", r"\bggo\b", r"\breticular opacit",
        r"\bpatchy opacit", r"\bhazy opacit", r"\bairspace disease\b",
        r"\binfiltrate\b",
    ],
    "Pulmonary fibrotic sequela": [
        r"\bfibroti", r"\bfibrosis\b", r"\bhoneycomb",
        r"\barchitectural distortion\b", r"\breticulation\b",
        r"\b(?:pulmonary |parenchymal |lung )scarr?ing\b",
        r"\b(?:parenchymal|pulmonary|lung) scar\b", r"\btraction bronchiectasis\b",
        r"\bfibrocystic\b", r"\binterstitial (?:lung )?disease\b",
        # CT-RATE phrasings: "(pleuro)parenchymal sequelae", "fibroatelectasis"
        r"\bsequela", r"\bfibro-?atelecta", r"\bpleuroparenchymal",
        r"\bparenchymal (?:band|sequela|change)", r"\blinear (?:scar|fibro|band)",
    ],
    "Pleural effusion": [
        r"\bpleural effusion", r"\bpleural fluid\b", r"\bhydrothorax\b",
        r"\beffusions?\b (?:in|within) the pleural", r"\bpleural (?:fluid|effusion)\b",
    ],
    "Mosaic attenuation pattern": [
        r"\bmosaic attenuation\b", r"\bmosaic pattern\b", r"\bmosaic perfusion\b",
        r"\bmosaicism\b",
    ],
    "Peribronchial thickening": [
        r"\bperibronchial (?:cuffing|thickenings?)\b", r"\bbronchial wall thickenings?\b",
        r"\bperibronchovascular (?:interstitial )?thickenings?\b",
        r"\bairway wall thickenings?\b", r"\bbronchial thickenings?\b",
        r"\bbronchial wall (?:is )?thickened\b",
    ],
    "Consolidation": [
        r"\bconsolidation", r"\bconsolidative\b", r"\bairspace consolidation\b",
        r"\bdense consolidation\b",
    ],
    "Bronchiectasis": [
        r"\bbronchiecta", r"\bdilated bronchi\b", r"\btraction bronchiectasis\b",
        r"\bcylindrical bronchiectasis\b",
    ],
    "Interlobular septal thickening": [
        # matches "interlobular septal thickening(s)" and "...septal thickness increase(s)"
        r"\binterlobular septal\b", r"\bseptal thickenings?\b", r"\bseptal thickness\b",
        r"\binterlobular septa\b", r"\bkerley\b", r"\bseptal lines\b",
        r"\bcrazy[- ]paving\b",
    ],
}

ABD_30 = {
    "Biliary Ductal Dilation": [
        r"\bbiliary (?:ductal |duct )?dilat", r"\bductal dilatation\b",
        r"\bdilat\w* (?:of (?:the )?)?(?:common bile duct|cbd|biliary (?:tree|ducts?)|intrahepatic ducts?|bile ducts?)\b",
        r"\bintrahepatic (?:biliary )?ductal dilat",
        r"\bdilated (?:cbd|common bile duct|bile ducts?|biliary (?:tree|ducts?)|intrahepatic ducts?)\b",
        r"\bprominent (?:cbd|common bile duct)\b",
    ],
    "Hepatomegaly": [
        r"\bhepatomegal", r"\benlarged liver\b", r"\bhepatic enlargement\b",
        r"\bliver (?:is )?enlarged\b", r"\benlargement of the liver\b",
    ],
    "Hepatic Steatosis": [
        r"\bhepatic steatosis\b", r"\bsteatosis\b", r"\bsteatotic\b",
        r"\bfatty liver\b", r"\bfatty (?:infiltration|change|metamorphosis|replacement) (?:of (?:the )?)?(?:liver|hepati)",
        r"\bhepatic (?:fatty|fat)\b",
        r"\bdiffuse(?:ly)? (?:low|decreased|hypo)\w* (?:hepatic |liver )?attenuation\b",
        r"\bdiffuse hepatic hypoattenuation\b",
    ],
    "Pancreatic Atrophy": [
        r"\bpancreatic atrophy\b", r"\batroph\w* (?:of the )?pancreas\b",
        r"\batrophic pancreas\b", r"\bpancreatic (?:fatty )?atroph",
        r"\bdiminutive pancreas\b",
        r"\bfatty (?:replacement|atrophy) of the pancreas\b",
    ],
    "Splenomegaly": [
        r"\bsplenomegal", r"\benlarged spleen\b", r"\bsplenic enlargement\b",
        r"\bspleen (?:is )?enlarged\b", r"\benlargement of the spleen\b",
    ],
    "Surgically Absent Gallbladder": [
        r"\bsurgically absent gallbladder\b", r"\bcholecystectomy\b",
        r"\bpost-?cholecystectomy\b", r"\bs/p cholecystectomy\b",
        r"\bgallbladder (?:is )?(?:surgically )?(?:absent|removed|resected)\b",
        r"\babsent gallbladder\b",
    ],
    "Gallstones": [
        r"\bgallstones?\b", r"\bcholelithiasis\b", r"\bcholelith",
        r"\bgallbladder (?:calculi|calculus|stones?)\b",
        r"\bcalculi (?:in|within) the gallbladder\b",
    ],
    "Renal Cyst": [
        r"\brenal cyst", r"\bcortical cyst", r"\bparapelvic cyst",
        r"\bbosniak\b", r"\bsimple (?:renal )?cyst",
        r"\bcyst(?:s|ic (?:lesion|focus|density))? (?:in|of|within) (?:the )?(?:kidney|kidneys|renal)\b",
    ],
    "Renal Hypodensity": [
        r"\brenal hypodensit", r"\bhypodense (?:renal|kidney) (?:lesion|focus|foci|lesions)\b",
        r"\b(?:hypodensity|hypodensities) (?:in|of|within) (?:the )?(?:kidney|kidneys|renal)\b",
        r"\blow[- ]attenuation (?:renal|kidney) (?:lesion|foci|focus)\b",
        r"\bhypoattenuating (?:renal|kidney) (?:lesion|focus|foci)\b",
    ],
    "Hydronephrosis": [
        r"\bhydronephros", r"\bhydroureteronephros", r"\bpelvicaliectasis\b",
        r"\bpelvocaliectasis\b", r"\bcaliectasis\b",
        r"\b(?:collecting system|renal pelvis|calyces) dilat",
        r"\bdilat\w* (?:of the )?(?:renal )?(?:collecting system|pelvis|calyces)\b",
        r"\bdilated (?:renal )?(?:pelvis|calyces|collecting system)\b",
    ],
    "Hiatal Hernia": [
        r"\bhiatal hernia\b", r"\bhiatus hernia\b", r"\bparaesophageal hernia\b",
    ],
    "Submucosal Edema": [
        r"\bsubmucosal edema\b", r"\bbowel wall edema\b", r"\bmural edema\b",
        r"\b(?:bowel|colonic|colon|small bowel|gastric) wall .{0,12}edema\b",
        r"\bedematous (?:bowel|colonic|small bowel) wall\b",
        r"\bsubmucosal (?:fluid|hypodensity|edema)\b",
    ],
    "Bowel Obstruction": [
        r"\bbowel obstruction\b", r"\bsmall bowel obstruction\b",
        r"\blarge bowel obstruction\b", r"\bsbo\b", r"\btransition point\b",
        r"\b(?:high-?grade|low-?grade|partial|complete) (?:bowel )?obstruction\b",
        r"\bobstructive (?:pattern|dilatation)\b",
    ],
    "Appendicitis": [
        r"\bappendicitis\b", r"\bperiappendiceal\b",
        r"\bappendiceal (?:wall )?(?:thickening|inflammation|dilat)\b",
        r"\b(?:inflamed|dilated|distended|enlarged|fluid-?filled) appendix\b",
    ],
    "Ascites": [
        r"\bascites\b", r"\bascitic\b",
        r"\bfree (?:intraperitoneal |intra-?abdominal |intraabdominal )?fluid\b",
        r"\b(?:intraperitoneal|peritoneal|perihepatic|perisplenic) fluid\b",
        r"\bfluid (?:in|within) the (?:abdomen|pelvis|peritoneal|paracolic|cul-?de-?sac|pouch of douglas)\b",
    ],
    "Free Air": [
        r"\bfree air\b", r"\bpneumoperitone", r"\bfree (?:intraperitoneal|intra-?abdominal) (?:air|gas)\b",
        r"\bextraluminal (?:air|gas)\b", r"\bfree (?:air|gas)\b",
    ],
    "Prostatomegaly": [
        r"\bprostatomegaly\b", r"\benlarged prostate\b",
        r"\bprostatic (?:enlargement|hypertrophy|hyperplasia)\b", r"\bbph\b",
        r"\bbenign prostatic (?:hypertrophy|hyperplasia)\b",
        r"\bprostate (?:is )?enlarged\b", r"\bprominent prostate\b",
    ],
    "Atherosclerosis": [
        r"\batherosclero", r"\batheromatous\b", r"\bvascular calcific",
        r"\barterial calcific", r"\bcalcified plaque",
        r"\baort\w*\b.{0,80}?\b(?:calcif|atheroma|atherosclerot|plaque)",
        r"\b(?:calcif|atheroma|atherosclerot|plaque)\w*\b.{0,80}?\b(?:aorta|iliac)",
    ],
    "Thrombosis": [
        r"\bthrombosis\b", r"\bthrombus\b", r"\bthrombi\b", r"\bthrombotic\b",
        r"\bthrombose", r"\b(?:dvt|deep (?:venous|vein) thrombosis)\b",
        r"\b(?:portal|mesenteric|splenic|renal|iliac|femoral|caval|ivc) vein thrombosis\b",
        r"\bvenous thrombosis\b", r"\bocclusive (?:thrombus|clot)\b",
    ],
    "Lymphadenopathy": [
        r"\blymphadenopathy\b", r"\badenopathy\b",
        r"\benlarged (?:retroperitoneal |mesenteric |para-?aortic |pelvic |inguinal |periportal )?lymph nodes?\b",
        r"\bpathologically enlarged (?:lymph )?nodes?\b",
        r"\bbulky (?:lymph )?nodes?\b",
        r"\bprominent (?:retroperitoneal |mesenteric )?lymph nodes?\b",
        r"\bnodal (?:enlargement|conglomerate|mass)\b",
    ],
    "Abdominal Aortic Aneurysm": [
        r"\babdominal aortic aneurysm\b", r"\baaa\b",
        r"\baortic aneurysm\b", r"\baneurysmal\b.{0,15}\baorta\b",
        r"\baorta\b.{0,15}\baneurysm", r"\binfrarenal (?:aortic )?aneurysm\b",
        r"\baneurysmal dilat\w* (?:of the )?(?:abdominal )?aorta\b",
    ],
    "Fracture": [
        r"\bfracture", r"\bfractured\b",
        r"\b(?:compression|wedge|burst|insufficiency) (?:fracture|deformit)",
        r"\bosseous fracture\b",
    ],
    "Osteopenia": [
        r"\bosteopenia\b", r"\bosteopenic\b", r"\bosteoporo",
        r"\bdemineraliz", r"\b(?:decreased|low) bone (?:density|mineraliz)",
        r"\bdiffuse(?:ly)? demineraliz",
    ],
    "Anasarca": [
        r"\banasarca\b", r"\bsubcutaneous edema\b", r"\bbody wall edema\b",
        r"\bdiffuse (?:soft tissue |subcutaneous |body wall )edema\b",
        r"\bgeneralized (?:body wall |subcutaneous )?edema\b", r"\bthird spacing\b",
    ],
    "Metastatic Disease": [
        r"\bmetasta", r"\bmets\b", r"\bcarcinomatosis\b",
        r"\bmetastatic (?:disease|deposits?|lesions?|spread)\b",
        r"\binnumerable (?:hepatic |osseous )?(?:lesions|metastases)\b",
    ],
    "Atelectasis": [
        r"\batelecta", r"\b(?:dependent|compressive|passive|basilar|basal) (?:atelectasis|atelectatic)\b",
    ],
    "Pleural Effusion": [
        r"\bpleural effusion", r"\bpleural fluid\b", r"\bhydrothorax\b",
        r"\bpleural (?:fluid|effusion)\b",
    ],
    "Cardiomegaly": [
        r"\bcardiomegaly\b", r"\benlarged (?:cardiac silhouette|heart)\b",
        r"\bcardiac (?:silhouette )?enlarge", r"\bheart (?:is )?enlarged\b",
    ],
    "Coronary Calcification": [
        r"\bcoronary (?:artery )?(?:wall )?calcific", r"\bcalcified coronary",
        r"\bcoronary calcium\b", r"\bcoronary atheroscleros",
        r"\bcoronary\b.{0,80}?\b(?:calcif|atheroma|atherosclerot|plaque)",
        r"\b(?:calcif|atheroma|atherosclerot|plaque)\w*\b.{0,80}?\bcoronary",
    ],
    "Aortic Valve Calcification": [
        r"\baortic valve calcific", r"\bcalcified aortic valve\b",
        r"\baortic valvular calcific", r"\bcalcification of the aortic valve\b",
        r"\baortic (?:valve|annular|leaflet) calcific",
    ],
}

# ---------------------------------------------------------------------------
# Context exclusions: a sentence matching one of these vetoes the finding for
# that sentence (wrong organ, boilerplate/recommendation templates).
# ---------------------------------------------------------------------------
_NODULE_BOILERPLATE = [
    r"follow-?up recommendation", r"lung-?rads", r"\[ln", r"not applicable",
    r"\bfleischner\b", r"derived from guidelines", r"do not necessarily apply",
    r"society \(radiology", r"\bguidelines\b",
]
_WRONG_ORGAN = [r"\bthyroid\b", r"\badrenal\b", r"\bhepatic\b", r"\bliver\b",
                r"\brenal\b", r"\bkidney\b", r"\bbreast\b"]
# hypothetical / study-limitation language (not an actual positive finding).
_HYPOTHETICAL = [r"\bif there\b", r"should be tailored", r"\blimited (?:for|in|by|due)\b",
                 r"evaluation (?:of|for) malignancy", r"\bif clinically\b"]

CHEST_EXCLUDE = {
    "Lung nodule": _WRONG_ORGAN + _NODULE_BOILERPLATE,
}
ABD_EXCLUDE = {
    "Lung nodule": _WRONG_ORGAN + _NODULE_BOILERPLATE,
    "Lymphadenopathy": [r"\bsub-?centimeter\b", r"\bsub-?cm\b"],
    "Metastatic Disease": _HYPOTHETICAL,
}


def get_rules(pool: str):
    """pool in {chest, abd} -> (presence_rules, exclude_rules)."""
    if pool == "chest":
        return CHEST_18, CHEST_EXCLUDE
    if pool == "abd":
        return ABD_30, ABD_EXCLUDE
    raise ValueError(pool)


assert len(CHEST_18) == 18, len(CHEST_18)
assert len(ABD_30) == 30, len(ABD_30)
