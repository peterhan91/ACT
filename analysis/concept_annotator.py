"""Map (label, concept) pairs to (color, citation_key, rationale).

Color scheme:
  BLUE   = concept describes a direct/definitional radiographic manifestation
           of the label
  PURPLE = concept describes a cited clinical / biological association but not
           a direct imaging manifestation (i.e., mechanistic confound)
  RED    = unsupported cohort confound / no disease-specific imaging link

Every BLUE / PURPLE call must carry a citation_key (resolved at render time
via citations.json) or "TAUTOLOGICAL" for direct/definitional matches that
need no literature. RED entries carry no citation.

Run:  python3 concept_annotator.py  (writes concept_annotations.json)
"""
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent

# ---- 1. Label → disease category ----------------------------------------
LABEL_TO_CATEGORY = {
    # ===== CT-RATE 14 (AUC>=0.75) =====
    "Pleural effusion":                 "DIRECT_PLEURAL",
    "Arterial wall calcification":      "DIRECT_VASCULAR_CALC",
    "Cardiomegaly":                     "DIRECT_CARDIOMEGALY",
    "Coronary artery wall calcification": "DIRECT_CORONARY_CALC",
    "Interlobular septal thickening":   "DIRECT_SEPTAL_THICKENING",
    "Consolidation":                    "DIRECT_CONSOLIDATION",
    "Pericardial effusion":             "DIRECT_PERICARDIAL",
    "Lung opacity":                     "DIRECT_LUNG_OPACITY",
    "Medical material":                 "DIRECT_MEDICAL_MATERIAL",
    "Peribronchial thickening":         "DIRECT_PERIBRONCHIAL",
    "Mosaic attenuation pattern":       "DIRECT_MOSAIC",
    "Emphysema":                        "DIRECT_EMPHYSEMA",
    "Hiatal hernia":                    "DIRECT_HIATAL_HERNIA",
    "Atelectasis":                      "DIRECT_ATELECTASIS",
    "Bronchiectasis":                   "BRONCHIECTASIS",
    "Lymphadenopathy":                  "DIRECT_LYMPHADENOPATHY",
    "Pulmonary fibrotic sequela":       "DIRECT_PULMONARY_SEQUELA",
    "Lung nodule":                      "DIRECT_LUNG_NODULE",
    # ===== INSPECT 69 (AUC>=0.75) =====
    "Pleurisy; pleural effusion":       "DIRECT_PLEURAL",
    "Ascites (non malignant)":          "GI_ASCITES",
    "Heart failure with preserved EF [Diastolic heart failure]": "HEART_FAILURE",
    "Heart failure with reduced EF [Systolic or combined heart failure]": "HEART_FAILURE",
    "Congestive heart failure (CHF) NOS": "HEART_FAILURE",
    "Hypertensive heart disease":       "HEART_FAILURE",
    "Hypertensive heart and/or renal disease": "HEART_FAILURE",
    "Heart valve disorders":            "HEART_VALVE",
    "Pulmonary congestion and hypostasis": "PULMONARY_EDEMA_CONGESTION",
    "Other pulmonary inflamation or edema": "PULMONARY_EDEMA_CONGESTION",
    "Fluid overload":                   "PULMONARY_EDEMA_CONGESTION",
    "Cancer within the respiratory system": "LUNG_CANCER",
    "Cancer of bronchus; lung":         "LUNG_CANCER",
    "Secondary malignancy of respiratory organs": "CANCER_GENERAL",
    "Secondary malignancy of bone":     "CANCER_GENERAL",
    "Secondary malignancy of lymph nodes": "CANCER_GENERAL",
    "Secondary malignant neoplasm":     "CANCER_GENERAL",
    "Cancer, suspected or other":       "CANCER_GENERAL",
    "Empyema and pneumothorax":         "EMPYEMA",
    "Septic shock":                     "SEPSIS_SHOCK",
    "Shock":                            "SEPSIS_SHOCK",
    "Hypotension NOS":                  "SEPSIS_SHOCK",
    "Hypotension":                      "SEPSIS_SHOCK",
    "Sepsis":                           "SEPSIS_SHOCK",
    "Respiratory failure":              "RESP_FAILURE",
    "Respiratory insufficiency":        "RESP_FAILURE",
    "Dependence on respirator [Ventilator] or supplemental oxygen": "RESP_FAILURE",
    "Disorders involving the immune mechanism": "ICU_GENERAL",
    "Asphyxia and hypoxemia":           "RESP_FAILURE",
    "Adjustment reaction":              "ICU_GENERAL",
    "Other abnormal glucose":           "METABOLIC",
    "Pneumonia":                        "PNEUMONIA",
    "Bacterial pneumonia":              "PNEUMONIA",
    "Pneumonitis due to inhalation of food or vomitus": "PNEUMONIA",
    "Anemia in neoplastic disease":     "CANCER_GENERAL",
    "Anemia in chronic kidney disease": "CKD",
    "Encephalopathy, not elsewhere classified": "ICU_GENERAL",
    "Delirium due to conditions classified elsewhere": "NEURO_COGNITIVE",
    "Delirium dementia and amnestic and other cognitive disorders": "NEURO_COGNITIVE",
    "Altered mental status":            "ICU_GENERAL",
    "Type 2 diabetes with renal manifestations": "T2D_RENAL",
    "Morbid obesity":                   "OBESITY",
    "Obesity":                          "OBESITY",
    "Overweight, obesity and other hyperalimentation": "OBESITY",
    "Paralytic ileus":                  "GI_ILEUS",
    "Other disorders of intestine":     "GI_ILEUS",
    "Other disorders of peritoneum":    "GI_ASCITES",
    "Complications of surgical and medical procedures": "GI_POSTOP",
    "severe protein-calorie malnutrition": "MALNUTRITION_POSTOP",
    "Protein-calorie malnutrition":     "MALNUTRITION_POSTOP",
    "Adult failure to thrive":          "MALNUTRITION_POSTOP",
    "Chronic bronchitis":               "BRONCHITIS",
    "Obstructive chronic bronchitis":   "BRONCHITIS",
    "Chronic airway obstruction":       "BRONCHITIS",
    "Emphysema":                        "DIRECT_EMPHYSEMA",  # CT-RATE
    "Bronchiectasis":                   "BRONCHIECTASIS",
    "Diaphragmatic hernia":             "DIRECT_HIATAL_HERNIA",
    "Abdominal hernia":                 "DIRECT_HIATAL_HERNIA",
    "End stage renal disease":          "CKD",
    "Chronic renal failure [CKD]":      "CKD",
    "Anemia of chronic disease":        "ANEMIA_CHRONIC",
    "Pulmonary collapse; interstitial and compensatory emphysema": "PULMONARY_COLLAPSE",
    "Hyperosmolality and/or hypernatremia": "METABOLIC",
    "Hyperpotassemia":                  "METABOLIC",
    "Alkalosis":                        "METABOLIC",
    "Acidosis":                         "METABOLIC",
    "Acid-base balance disorder":       "METABOLIC",
    "Osteoporosis NOS":                 "OSTEOPOROSIS",
    "Other ill-defined and unknown causes of morbidity and mortality": "ICU_GENERAL",
    "Hyperosmolality and/or hyponatremia": "METABOLIC",
    # explicit duplicate keys that may exist:
    "Hyperpotassemia ":                 "METABOLIC",
}


# ---- 2. (Category, Motif) → (color, citation_key) decision table --------
# Citation keys are resolved at render time via citations.json.
# Use TAUTOLOGICAL only for direct/definitional radiology terms that should
# appear in figures as "well known*".
B, P, R = "BLUE", "PURPLE", "RED"
DECISIONS: dict[tuple[str, str], tuple[str, str]] = {
    # ===== DIRECT_PLEURAL (Pleural effusion, Pleurisy; pleural effusion) =====
    ("DIRECT_PLEURAL", "M_PLEURAL_EFFUSION"):       (B, "TAUTOLOGICAL"),
    ("DIRECT_PLEURAL", "M_PLEURAL_EFFUSION_ATEL"):  (B, "TAUTOLOGICAL"),
    ("DIRECT_PLEURAL", "M_PERICARDIAL_EFFUSION"):   (P, "Porcel2013_pleural"),
    ("DIRECT_PLEURAL", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Light2002_pleural"),
    ("DIRECT_PLEURAL", "M_LOBAR_CONSOLIDATION"):    (P, "Light2002_pleural"),

    # ===== DIRECT_PERICARDIAL =====
    # Pericardial-effusion patients often have concurrent pleural effusion +
    # passive atelectasis (shared fluid-overload aetiology) — Porcel 2013
    # documents the pleural-effusion side of this concurrent-serous-cavity
    # presentation; Adler 2015 ESC covers the pericardial side.
    ("DIRECT_PERICARDIAL", "M_PERICARDIAL_EFFUSION"): (B, "TAUTOLOGICAL"),
    ("DIRECT_PERICARDIAL", "M_PLEURAL_EFFUSION"):    (P, "Porcel2013_pleural"),
    ("DIRECT_PERICARDIAL", "M_PLEURAL_EFFUSION_ATEL"): (P, "Porcel2013_pleural"),
    ("DIRECT_PERICARDIAL", "M_ATELECTASIS"):         (P, "Porcel2013_pleural"),
    ("DIRECT_PERICARDIAL", "M_UNCLASSIFIED"):        (R, ""),

    # ===== DIRECT_VASCULAR_CALC (Arterial wall calcification) =====
    ("DIRECT_VASCULAR_CALC", "M_AORTIC_CA_THORACIC_COR"): (B, "TAUTOLOGICAL"),
    ("DIRECT_VASCULAR_CALC", "M_AORTIC_CA_ABDOMINAL"):    (B, "TAUTOLOGICAL"),
    ("DIRECT_VASCULAR_CALC", "M_AORTIC_CA_GENERIC"):      (B, "TAUTOLOGICAL"),
    ("DIRECT_VASCULAR_CALC", "M_AORTIC_CA_MILD"):         (B, "TAUTOLOGICAL"),

    # ===== DIRECT_CORONARY_CALC =====
    ("DIRECT_CORONARY_CALC", "M_AORTIC_CA_THORACIC_COR"): (B, "TAUTOLOGICAL"),
    ("DIRECT_CORONARY_CALC", "M_AORTIC_CA_GENERIC"):      (P, "Allison2004_crossbed"),
    ("DIRECT_CORONARY_CALC", "M_AORTIC_CA_ABDOMINAL"):    (P, "Allison2004_crossbed"),
    ("DIRECT_CORONARY_CALC", "M_AORTIC_CA_MILD"):         (P, "Allison2004_crossbed"),

    # ===== DIRECT_CARDIOMEGALY =====
    ("DIRECT_CARDIOMEGALY", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("DIRECT_CARDIOMEGALY", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("DIRECT_CARDIOMEGALY", "M_AORTIC_CA_MILD"):         (R, ""),
    ("DIRECT_CARDIOMEGALY", "M_AORTIC_CA_ABDOMINAL"):    (R, ""),
    ("DIRECT_CARDIOMEGALY", "M_PERICARDIAL_EFFUSION"):   (P, "Adler2015_pericardial_ESC"),

    # ===== DIRECT_CONSOLIDATION =====
    ("DIRECT_CONSOLIDATION", "M_GGO_CONSOLIDATION_DIFFUSE"): (B, "TAUTOLOGICAL"),
    ("DIRECT_CONSOLIDATION", "M_LOBAR_CONSOLIDATION"):       (B, "TAUTOLOGICAL"),

    # ===== DIRECT_LUNG_OPACITY =====
    ("DIRECT_LUNG_OPACITY", "M_GGO_CONSOLIDATION_DIFFUSE"): (B, "TAUTOLOGICAL"),
    ("DIRECT_LUNG_OPACITY", "M_LOBAR_CONSOLIDATION"):       (B, "TAUTOLOGICAL"),

    # ===== DIRECT_SEPTAL_THICKENING =====
    # Crazy-paving DOES include septal thickening; concepts mentioning
    # "crazy paving" or "interlobular septal thickening" are blue, generic
    # GGO/consolidation patterns without explicit septal mention are purple.
    ("DIRECT_SEPTAL_THICKENING", "M_GGO_CONSOLIDATION_DIFFUSE"): (R, ""),

    # ===== DIRECT_PERIBRONCHIAL =====
    ("DIRECT_PERIBRONCHIAL", "M_BRONCHIECTASIS"):       (P, "Cantin2009_bronchiectasis_CT"),
    ("DIRECT_PERIBRONCHIAL", "M_POSTSURGICAL_THORACIC"): (R, ""),
    ("DIRECT_PERIBRONCHIAL", "M_THORACOTOMY_RIB"):       (R, ""),
    ("DIRECT_PERIBRONCHIAL", "M_LOBECTOMY"):             (R, ""),

    # ===== DIRECT_MOSAIC =====
    # Mosaic attenuation has no direct biologic link to aortic Ca²⁺.
    ("DIRECT_MOSAIC", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("DIRECT_MOSAIC", "M_AORTIC_CA_MILD"):         (R, ""),
    ("DIRECT_MOSAIC", "M_AORTIC_CA_GENERIC"):      (R, ""),

    # ===== DIRECT_EMPHYSEMA (CT-RATE) =====
    # Concepts are about liver lesions / hepatic masses — confound.
    ("DIRECT_EMPHYSEMA", "M_LIVER_MASS_LESION"):    (R, ""),
    ("DIRECT_EMPHYSEMA", "M_BILIARY_DRAIN_STENT"):  (R, ""),
    ("DIRECT_EMPHYSEMA", "M_BILE_DUCT_DILATION"):   (R, ""),
    ("DIRECT_EMPHYSEMA", "M_POSTSURGICAL_UPPER_GI"): (R, ""),

    # ===== DIRECT_HIATAL_HERNIA / Diaphragmatic / Abdominal =====
    ("DIRECT_HIATAL_HERNIA", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("DIRECT_HIATAL_HERNIA", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("DIRECT_HIATAL_HERNIA", "M_AORTIC_CA_ABDOMINAL"):    (R, ""),

    # ===== DIRECT_ATELECTASIS =====
    ("DIRECT_ATELECTASIS", "M_ATELECTASIS"):           (B, "TAUTOLOGICAL"),
    ("DIRECT_ATELECTASIS", "M_PLEURAL_EFFUSION_ATEL"): (B, "TAUTOLOGICAL"),
    ("DIRECT_ATELECTASIS", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("DIRECT_ATELECTASIS", "M_AORTIC_CA_GENERIC"):      (R, ""),

    # ===== DIRECT_MEDICAL_MATERIAL =====
    ("DIRECT_MEDICAL_MATERIAL", "M_POSTSURGICAL_UPPER_GI"): (R, ""),
    ("DIRECT_MEDICAL_MATERIAL", "M_POSTSURGICAL_THORACIC"): (R, ""),
    ("DIRECT_MEDICAL_MATERIAL", "M_BILIARY_DRAIN_STENT"):   (B, "TAUTOLOGICAL"),
    ("DIRECT_MEDICAL_MATERIAL", "M_SURGICAL_DRAIN"):        (B, "TAUTOLOGICAL"),

    # ===== EMPYEMA (Empyema and pneumothorax) =====
    ("EMPYEMA", "M_PLEURAL_EFFUSION_ATEL"): (B, "Davies2010_BTS"),
    ("EMPYEMA", "M_PLEURAL_EFFUSION"):      (B, "Davies2010_BTS"),
    ("EMPYEMA", "M_LOBAR_CONSOLIDATION"):   (P, "Davies2010_BTS"),
    ("EMPYEMA", "M_BILIARY_DRAIN_STENT"):   (R, ""),
    ("EMPYEMA", "M_SURGICAL_DRAIN"):        (P, "Davies2010_BTS"),

    # ===== HEART_FAILURE =====
    ("HEART_FAILURE", "M_AORTIC_CA_THORACIC_COR"): (P, "Sharma2017_CAC_HFpEF"),
    ("HEART_FAILURE", "M_AORTIC_CA_GENERIC"):      (P, "Sharma2017_CAC_HFpEF"),
    ("HEART_FAILURE", "M_AORTIC_CA_ABDOMINAL"):    (P, "Sharma2017_CAC_HFpEF"),
    ("HEART_FAILURE", "M_AORTIC_CA_MILD"):         (P, "Sharma2017_CAC_HFpEF"),
    ("HEART_FAILURE", "M_PERICARDIAL_EFFUSION"):   (P, "Adler2015_pericardial_ESC"),
    ("HEART_FAILURE", "M_PLEURAL_EFFUSION"):       (B, "Light2002_pleural"),
    ("HEART_FAILURE", "M_PLEURAL_EFFUSION_ATEL"):  (B, "Light2002_pleural"),
    ("HEART_FAILURE", "M_IVC_HEPATIC_CONGESTION"): (B, "Clark2015_CT_HF"),

    # ===== HEART_VALVE =====
    ("HEART_VALVE", "M_AORTIC_CA_THORACIC_COR"): (P, "Mohler2001_valve"),
    ("HEART_VALVE", "M_AORTIC_CA_GENERIC"):      (P, "Mohler2001_valve"),

    # ===== PULMONARY_EDEMA_CONGESTION =====
    ("PULMONARY_EDEMA_CONGESTION", "M_PERICARDIAL_EFFUSION"):   (P, "Adler2015_pericardial_ESC"),
    ("PULMONARY_EDEMA_CONGESTION", "M_IVC_HEPATIC_CONGESTION"): (P, "Clark2015_CT_HF"),
    ("PULMONARY_EDEMA_CONGESTION", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("PULMONARY_EDEMA_CONGESTION", "M_VICARIOUS_EXCRETION"):    (R, ""),
    ("PULMONARY_EDEMA_CONGESTION", "M_GGO_CONSOLIDATION_DIFFUSE"): (B, "Storto1995_PE"),

    # ===== LUNG_CANCER =====
    ("LUNG_CANCER", "M_LUNG_MASS_TUMOR"):          (B, "TAUTOLOGICAL"),
    ("LUNG_CANCER", "M_LOBAR_CONSOLIDATION"):      (P, "Valvani2019_postobstructive"),
    ("LUNG_CANCER", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Valvani2019_postobstructive"),
    ("LUNG_CANCER", "M_BRONCHIECTASIS"):           (R, ""),
    ("LUNG_CANCER", "M_SURGICAL_DRAIN"):           (R, ""),
    ("LUNG_CANCER", "M_BILIARY_DRAIN_STENT"):      (R, ""),

    # ===== SEPSIS_SHOCK =====
    # Sepsis classically produces distributive shock with a COLLAPSED IVC;
    # the model's "distended IVC + contrast reflux" association in septic-
    # shock cases is therefore a confounded sign (likely reflecting mixed
    # cardiogenic component or volume resuscitation). Long 2017 documents
    # the unreliability of IVC US in critical care, framing this as a known-
    # weak/non-canonical association; keep it red in a strict radiology audit.
    ("SEPSIS_SHOCK", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("SEPSIS_SHOCK", "M_PERIHEPATIC_EFFUSION"):   (R, ""),
    ("SEPSIS_SHOCK", "M_PERICARDIAL_EFFUSION"):   (R, ""),
    ("SEPSIS_SHOCK", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("SEPSIS_SHOCK", "M_AORTIC_CA_ABDOMINAL"):    (R, ""),
    ("SEPSIS_SHOCK", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("SEPSIS_SHOCK", "M_PLEURAL_EFFUSION"):       (R, ""),

    # ===== RESP_FAILURE =====
    ("RESP_FAILURE", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Sheard2012_ARDS"),
    ("RESP_FAILURE", "M_LOBAR_CONSOLIDATION"):       (P, "Sheard2012_ARDS"),
    ("RESP_FAILURE", "M_BILE_DUCT_DILATION"):        (R, ""),
    ("RESP_FAILURE", "M_BILIARY_DRAIN_STENT"):       (R, ""),
    ("RESP_FAILURE", "M_AORTIC_CA_GENERIC"):         (R, ""),
    ("RESP_FAILURE", "M_AORTIC_CA_ABDOMINAL"):       (R, ""),

    # ===== PNEUMONIA / PNEUMONIA_SEPSIS =====
    ("PNEUMONIA", "M_GGO_CONSOLIDATION_DIFFUSE"): (B, "Franquet2018_pneumonia"),
    ("PNEUMONIA", "M_LOBAR_CONSOLIDATION"):       (B, "Franquet2018_pneumonia"),
    ("PNEUMONIA", "M_PLEURAL_EFFUSION"):          (B, "Sahn2007_parapneumonic"),
    ("PNEUMONIA_SEPSIS", "M_GGO_CONSOLIDATION_DIFFUSE"): (B, "Franquet2018_pneumonia"),
    ("PNEUMONIA_SEPSIS", "M_LOBAR_CONSOLIDATION"):       (B, "Franquet2018_pneumonia"),

    # ===== CKD / ESRD =====
    # Goodman 2000 NEJM: seminal CT-imaging documentation of extensive
    # coronary-artery calcification in young ESRD dialysis patients tied to
    # dialysis vintage + Ca×P — the canonical imaging-grounded CKD-MBD cite.
    ("CKD", "M_AORTIC_CA_THORACIC_COR"): (P, "Goodman2000_ESRD_CAC"),
    ("CKD", "M_AORTIC_CA_GENERIC"):      (P, "Goodman2000_ESRD_CAC"),
    ("CKD", "M_AORTIC_CA_ABDOMINAL"):    (P, "Goodman2000_ESRD_CAC"),
    ("CKD", "M_AORTIC_CA_MILD"):         (P, "Goodman2000_ESRD_CAC"),
    ("CKD", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("CKD", "M_PLEURAL_EFFUSION"):       (R, ""),

    # ===== T2D_RENAL =====
    # Lehto 1996 ATVB: T2D-specific medial (Mönckeberg) calcification
    # predicts CV/CHD/stroke mortality — the canonical diabetes-specific
    # vascular-Ca²⁺ paper, fitting the audit's "intimal+medial plaque"
    # phrasing in T2D-renal patients better than the prior London 2003 NDT
    # (ESRD-only) citation.
    ("T2D_RENAL", "M_AORTIC_CA_THORACIC_COR"): (P, "Lehto1996_T2D_medial"),
    ("T2D_RENAL", "M_AORTIC_CA_GENERIC"):      (P, "Lehto1996_T2D_medial"),
    ("T2D_RENAL", "M_AORTIC_CA_ABDOMINAL"):    (P, "Lehto1996_T2D_medial"),

    # ===== OBESITY =====
    ("OBESITY", "M_AORTIC_CA_MILD"):     (P, "Kronmal2007_MESA_CAC_riskfactors"),
    ("OBESITY", "M_AORTIC_CA_THORACIC_COR"): (P, "Kronmal2007_MESA_CAC_riskfactors"),
    ("OBESITY", "M_AORTIC_CA_GENERIC"):  (P, "Kronmal2007_MESA_CAC_riskfactors"),

    # ===== METABOLIC (electrolyte / acid-base / glucose) =====
    ("METABOLIC", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("METABOLIC", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("METABOLIC", "M_PERICARDIAL_EFFUSION"):   (R, ""),
    ("METABOLIC", "M_PERIHEPATIC_EFFUSION"):   (R, ""),
    ("METABOLIC", "M_VICARIOUS_EXCRETION"):    (R, ""),
    ("METABOLIC", "M_PANCREATITIS"):           (R, ""),
    ("METABOLIC", "M_GGO_CONSOLIDATION_DIFFUSE"): (R, ""),
    ("METABOLIC", "M_LOBAR_CONSOLIDATION"):    (R, ""),
    ("METABOLIC", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("METABOLIC", "M_AORTIC_CA_ABDOMINAL"):    (R, ""),
    ("METABOLIC", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("METABOLIC", "M_UNCLASSIFIED"):           (R, ""),

    # ===== NEURO_COGNITIVE =====
    ("NEURO_COGNITIVE", "M_AORTIC_CA_THORACIC_COR"): (P, "Bos2015_dementia"),
    ("NEURO_COGNITIVE", "M_AORTIC_CA_GENERIC"):      (P, "Bos2015_dementia"),
    ("NEURO_COGNITIVE", "M_AORTIC_CA_ABDOMINAL"):    (P, "Bos2015_dementia"),
    ("NEURO_COGNITIVE", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("NEURO_COGNITIVE", "M_VICARIOUS_EXCRETION"):    (R, ""),

    # ===== ANEMIA_CHRONIC =====
    ("ANEMIA_CHRONIC", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("ANEMIA_CHRONIC", "M_BILIARY_DRAIN_STENT"):    (R, ""),
    ("ANEMIA_CHRONIC", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("ANEMIA_CHRONIC", "M_PLEURAL_EFFUSION_ATEL"):  (R, ""),
    ("ANEMIA_CHRONIC", "M_PLEURAL_EFFUSION"):       (R, ""),

    # ===== CANCER_GENERAL (anemia in neoplastic) =====
    ("CANCER_GENERAL", "M_LOBAR_CONSOLIDATION"):    (R, ""),
    ("CANCER_GENERAL", "M_GGO_CONSOLIDATION_DIFFUSE"): (R, ""),
    ("CANCER_GENERAL", "M_PLEURAL_EFFUSION_ATEL"):  (R, ""),
    ("CANCER_GENERAL", "M_SURGICAL_DRAIN"):         (R, ""),

    # ===== BRONCHITIS / chronic airway =====
    ("BRONCHITIS", "M_AORTIC_CA_THORACIC_COR"): (P, "Sin2003_COPD"),
    ("BRONCHITIS", "M_AORTIC_CA_GENERIC"):      (P, "Sin2003_COPD"),
    ("BRONCHITIS", "M_AORTIC_CA_ABDOMINAL"):    (P, "Sin2003_COPD"),
    ("BRONCHITIS", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("BRONCHITIS", "M_BILIARY_DRAIN_STENT"):    (R, ""),
    ("BRONCHITIS", "M_POSTSURGICAL_UPPER_GI"):  (R, ""),

    # ===== BRONCHIECTASIS =====
    # The structural lesion of bronchiectasis is bronchial dilation
    # (tautological match → green). Consolidation, GGO, tree-in-bud, and
    # peribronchial-thickening patterns in a bronchiectasis-labelled
    # patient reflect superimposed infectious exacerbation / mucus
    # impaction rather than the disease itself — a known clinical-imaging
    # association (purple), per Cantin 2009 AJR pictorial review.
    ("BRONCHIECTASIS", "M_LOBAR_CONSOLIDATION"):    (P, "Cantin2009_bronchiectasis_CT"),
    ("BRONCHIECTASIS", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Cantin2009_bronchiectasis_CT"),
    ("BRONCHIECTASIS", "M_BRONCHIECTASIS"):         (B, "TAUTOLOGICAL"),
    ("BRONCHIECTASIS", "M_LUNG_NODULE"):            (R, ""),
    ("BRONCHIECTASIS", "M_NONPULMONARY_NODULE"):    (R, ""),

    # ===== DIRECT_LYMPHADENOPATHY =====
    ("DIRECT_LYMPHADENOPATHY", "M_LYMPHADENOPATHY"): (B, "TAUTOLOGICAL"),
    ("DIRECT_LYMPHADENOPATHY", "M_BILE_DUCT_DILATION"): (R, ""),
    ("DIRECT_LYMPHADENOPATHY", "M_BILIARY_DRAIN_STENT"): (R, ""),
    ("DIRECT_LYMPHADENOPATHY", "M_LIVER_MASS_LESION"): (R, ""),

    # ===== DIRECT_PULMONARY_SEQUELA =====
    ("DIRECT_PULMONARY_SEQUELA", "M_PULMONARY_SEQUELA"): (B, "TAUTOLOGICAL"),
    ("DIRECT_PULMONARY_SEQUELA", "M_BRONCHIECTASIS"):    (R, ""),
    ("DIRECT_PULMONARY_SEQUELA", "M_LUNG_NODULE"):       (R, ""),
    ("DIRECT_PULMONARY_SEQUELA", "M_NONPULMONARY_NODULE"): (R, ""),
    ("DIRECT_PULMONARY_SEQUELA", "M_LIVER_MASS_LESION"): (R, ""),
    ("DIRECT_PULMONARY_SEQUELA", "M_AORTIC_CA_GENERIC"): (R, ""),
    ("DIRECT_PULMONARY_SEQUELA", "M_ATELECTASIS"):       (R, ""),

    # ===== DIRECT_LUNG_NODULE =====
    ("DIRECT_LUNG_NODULE", "M_LUNG_NODULE"):          (B, "TAUTOLOGICAL"),
    ("DIRECT_LUNG_NODULE", "M_NONPULMONARY_NODULE"):  (R, ""),
    ("DIRECT_LUNG_NODULE", "M_PULMONARY_SEQUELA"):    (B, "TAUTOLOGICAL"),

    # ===== GI_ASCITES =====
    ("GI_ASCITES", "M_ASCITES_PERITONEAL"):     (B, "TAUTOLOGICAL"),
    ("GI_ASCITES", "M_PLEURAL_EFFUSION"):       (P, "Krok2014_hydrothorax"),
    ("GI_ASCITES", "M_PLEURAL_EFFUSION_ATEL"):  (P, "Krok2014_hydrothorax"),
    ("GI_ASCITES", "M_BOWEL_DISTENSION_FLUID"): (R, ""),

    # ===== GI_ILEUS =====
    ("GI_ILEUS", "M_BOWEL_DISTENSION_FLUID"): (B, "Silva2009_ileus"),
    ("GI_ILEUS", "M_ASCITES_PERITONEAL"):     (R, ""),

    # ===== GI_POSTOP (Complications of surgical/medical procedures) =====
    ("GI_POSTOP", "M_BOWEL_DISTENSION_FLUID"): (B, "Silva2009_ileus"),
    ("GI_POSTOP", "M_ASCITES_PERITONEAL"):     (R, ""),
    ("GI_POSTOP", "M_PERICARDIAL_EFFUSION"):   (R, ""),

    # ===== MALNUTRITION_POSTOP =====
    ("MALNUTRITION_POSTOP", "M_BILIARY_DRAIN_STENT"):    (R, ""),
    ("MALNUTRITION_POSTOP", "M_POSTSURGICAL_UPPER_GI"):  (R, ""),
    ("MALNUTRITION_POSTOP", "M_SURGICAL_DRAIN"):         (R, ""),
    ("MALNUTRITION_POSTOP", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("MALNUTRITION_POSTOP", "M_PLEURAL_EFFUSION_ATEL"):  (R, ""),
    ("MALNUTRITION_POSTOP", "M_IVC_HEPATIC_CONGESTION"): (R, ""),

    # ===== PULMONARY_COLLAPSE (interstitial / compensatory emphysema) =====
    ("PULMONARY_COLLAPSE", "M_PLEURAL_EFFUSION_ATEL"): (B, "Kuhlman1997_pleuralCT"),
    ("PULMONARY_COLLAPSE", "M_ATELECTASIS"):           (B, "Kuhlman1997_pleuralCT"),
    ("PULMONARY_COLLAPSE", "M_PLEURAL_EFFUSION"):      (B, "Kuhlman1997_pleuralCT"),

    # ===== OSTEOPOROSIS =====
    ("OSTEOPOROSIS", "M_AORTIC_CA_THORACIC_COR"): (P, "Schulz2004_osteoporosis"),
    ("OSTEOPOROSIS", "M_AORTIC_CA_ABDOMINAL"):    (P, "Schulz2004_osteoporosis"),
    ("OSTEOPOROSIS", "M_AORTIC_CA_GENERIC"):      (P, "Schulz2004_osteoporosis"),

    # ===== ICU_GENERAL (Other ill-defined causes of morbidity/mortality) =====
    ("ICU_GENERAL", "M_PLEURAL_EFFUSION_ATEL"): (R, ""),
    ("ICU_GENERAL", "M_PLEURAL_EFFUSION"):      (R, ""),
    ("ICU_GENERAL", "M_LOBAR_CONSOLIDATION"):   (R, ""),
    ("ICU_GENERAL", "M_GGO_CONSOLIDATION_DIFFUSE"): (R, ""),
    ("ICU_GENERAL", "M_IVC_HEPATIC_CONGESTION"): (R, ""),
    ("ICU_GENERAL", "M_AORTIC_CA_THORACIC_COR"): (R, ""),
    ("ICU_GENERAL", "M_AORTIC_CA_GENERIC"):      (R, ""),
    ("ICU_GENERAL", "M_AORTIC_CA_ABDOMINAL"):    (R, ""),
    ("ICU_GENERAL", "M_AORTIC_CA_MILD"):         (R, ""),
    ("ICU_GENERAL", "M_PERICARDIAL_EFFUSION"):   (R, ""),
    ("ICU_GENERAL", "M_PERIHEPATIC_EFFUSION"):   (R, ""),
    ("ICU_GENERAL", "M_ASCITES_PERITONEAL"):     (R, ""),
    ("ICU_GENERAL", "M_BILE_DUCT_DILATION"):     (R, ""),
    ("ICU_GENERAL", "M_BILIARY_DRAIN_STENT"):    (R, ""),
    ("ICU_GENERAL", "M_SURGICAL_DRAIN"):         (R, ""),
    ("ICU_GENERAL", "M_POSTSURGICAL_UPPER_GI"):  (R, ""),
    ("ICU_GENERAL", "M_POSTSURGICAL_THORACIC"):  (R, ""),
    ("ICU_GENERAL", "M_BOWEL_DISTENSION_FLUID"): (R, ""),
    ("ICU_GENERAL", "M_LYMPHADENOPATHY"):        (R, ""),
    ("ICU_GENERAL", "M_LUNG_NODULE"):            (R, ""),
    ("ICU_GENERAL", "M_LIVER_MASS_LESION"):      (R, ""),
    ("ICU_GENERAL", "M_VICARIOUS_EXCRETION"):    (R, ""),
    ("ICU_GENERAL", "M_ATELECTASIS"):            (R, ""),
    ("ICU_GENERAL", "M_UNCLASSIFIED"):           (R, ""),

    # ===== fill-in for unmapped pairs =====
    # Lung cancer × surgery/drains/fluid collections: treatment or cohort axis,
    # not a specific current imaging manifestation in these top concepts.
    ("LUNG_CANCER", "M_POSTSURGICAL_UPPER_GI"): (R, ""),
    ("LUNG_CANCER", "M_POSTSURGICAL_THORACIC"): (R, ""),
    ("LUNG_CANCER", "M_PERIHEPATIC_EFFUSION"):  (R, ""),
    # Emphysema (CT-RATE) × lung mass: emphysema/smoking and lung cancer share
    # smoking aetiology — co-morbidity, not a direct emphysema imaging feature.
    ("DIRECT_EMPHYSEMA", "M_LUNG_MASS_TUMOR"):  (P, "DeTorres2007_emphysema_lung_cancer"),
    # Empyema × diffuse GGO+consolidation: parapneumonic empyema commonly has
    # adjacent pneumonic consolidation, but consolidation is not the pleural
    # space diagnosis itself.
    ("EMPYEMA", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Sahn2007_parapneumonic"),
    # Pulmonary edema/congestion × perihepatic fluid: specific only for volume
    # overload, but not a pulmonary-edema imaging finding in this audit.
    ("PULMONARY_EDEMA_CONGESTION", "M_PERIHEPATIC_EFFUSION"): (R, ""),
    # Neuro cognitive × bile duct: cohort confound (sick inpatient).
    ("NEURO_COGNITIVE", "M_BILE_DUCT_DILATION"): (R, ""),
    # Pneumonia × bile duct: cohort confound.
    ("PNEUMONIA", "M_BILE_DUCT_DILATION"): (R, ""),
    # SEPSIS_SHOCK × diffuse GGO+consolidation: sepsis is a major ARDS driver,
    # but the lung opacities are a related acute-lung-injury confound rather
    # than a direct hemodynamic manifestation of shock.
    ("SEPSIS_SHOCK", "M_GGO_CONSOLIDATION_DIFFUSE"): (P, "Cusack2023_sepsis_ARDS"),
    # ICU_GENERAL × diffuse GGO+consolidation: cohort confound on the
    # sick-inpatient axis — these labels do not have a specific imaging
    # phenotype, so GGO/consolidation here is co-morbid lung injury, not a
    # direct radiographic finding of the label.
    ("ICU_GENERAL", "M_GGO_CONSOLIDATION_DIFFUSE"): (R, ""),
    # CANCER_GENERAL × post-surgical / perihepatic-effusion: drain / fluid
    # markers reflect cohort surgical history, not the cancer label itself.
    ("CANCER_GENERAL", "M_POSTSURGICAL_UPPER_GI"): (R, ""),
    ("CANCER_GENERAL", "M_POSTSURGICAL_THORACIC"): (R, ""),
    ("CANCER_GENERAL", "M_PERIHEPATIC_EFFUSION"):  (R, ""),
}


# ---- 3. Assignment driver -----------------------------------------------
PULMONARY_LOCATION_RE = re.compile(
    r"\b(lung|lungs|lobe|lobes|pleura|pleural|fissure|pulmonary|"
    r"apical|apicoposterior|anterior segment|posterior segment|"
    r"superior segment|basal segment|lingular|middle lobe)\b"
)
MILD_CALC_RE = re.compile(r"\b(mild|minimal|millimetric|millimetric-sized)\b")


def _category_for_label(label: str) -> str:
    """Conservative category fallback for full INSPECT figures.

    Labels not explicitly mapped are treated as nonspecific cohort labels so
    their top concepts default to red rather than unsupported green/purple.
    """
    if label in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[label]
    l = label.lower()
    if "pleural" in l and "effusion" in l:
        return "DIRECT_PLEURAL"
    if "pericard" in l:
        return "DIRECT_PERICARDIAL"
    if "pneumonia" in l or "pneumonitis" in l:
        return "PNEUMONIA"
    if "respiratory failure" in l or "respiratory insufficiency" in l:
        return "RESP_FAILURE"
    if "chronic kidney" in l or "end stage renal" in l:
        return "CKD"
    if "diabetes" in l:
        return "T2D_RENAL"
    if "obesity" in l or "hyperalimentation" in l:
        return "OBESITY"
    if "coronary atherosclerosis" in l:
        return "DIRECT_CORONARY_CALC"
    if "dementia" in l:
        return "NEURO_COGNITIVE"
    if "electrolyte" in l or "potassemia" in l or "osmolality" in l:
        return "METABOLIC"
    if "acid-base" in l or "acidosis" in l or "alkalosis" in l:
        return "METABOLIC"
    return "ICU_GENERAL"
NONPULMONARY_LOCATION_RE = re.compile(
    r"\b(abdomen|abdominal|intercostal|diaphragm|hepatic|liver|"
    r"pancreatic|pancreas|bile duct|biliary|common duct)\b"
)


def _infer_motif(label: str, concept: str) -> str:
    """Small deterministic motif backfill for current top-10 labels."""
    c = concept.lower()
    is_pulmonary = bool(PULMONARY_LOCATION_RE.search(c))
    is_nonpulmonary = bool(NONPULMONARY_LOCATION_RE.search(c))

    if (
        "pleural effusion" in c
        or "pleural effusions" in c
        or ("effusion" in c and "pleural" in c)
    ):
        if "atelect" in c or "consolidat" in c:
            return "M_PLEURAL_EFFUSION_ATEL"
        return "M_PLEURAL_EFFUSION"
    if "pericardial effusion" in c or ("effusion" in c and "pericardial" in c):
        return "M_PERICARDIAL_EFFUSION"
    if (
        "ivc" in c
        or "inferior vena cava" in c
        or "hepatic venous" in c
        or "hepatic veins" in c
        or "venous congestion" in c
        or "poor cardiac output" in c
    ):
        return "M_IVC_HEPATIC_CONGESTION"
    if "lymphadenopathy" in c or "lymph node" in c or "lymph nodes" in c:
        return "M_LYMPHADENOPATHY"
    if "biliary stent" in c or "bile duct stent" in c:
        return "M_BILIARY_DRAIN_STENT"
    if "bile duct" in c or "biliary duct" in c or "common duct" in c:
        return "M_BILE_DUCT_DILATION"
    if "cholangiocarcinoma" in c or "hepatic lobe" in c or "liver" in c:
        return "M_LIVER_MASS_LESION"
    if "vicarious excretion" in c:
        return "M_VICARIOUS_EXCRETION"
    if "perihepatic effusion" in c or "perihepatic fluid" in c:
        return "M_PERIHEPATIC_EFFUSION"
    if (
        "ascites" in c
        or "intraperitoneal fluid" in c
        or "peritoneal fluid" in c
        or "intra-abdominal effusion" in c
        or "free fluid" in c
    ):
        return "M_ASCITES_PERITONEAL"
    if (
        "drainage catheter" in c
        or "surgical drain" in c
        or "pigtail drain" in c
        or "percutaneous drain" in c
        or "cholecystostomy tube" in c
        or "multiple mediastinal and bilateral drains" in c
    ):
        return "M_SURGICAL_DRAIN"
    if (
        "post-sternotomy" in c
        or "sternotomy" in c
        or "thoracic aortic stent" in c
        or "ivor-lewis" in c
        or "lobectomy" in c
        or "pneumonectomy" in c
    ):
        return "M_POSTSURGICAL_THORACIC"
    if (
        "postsurgical" in c
        or "post surgical" in c
        or "postoperative" in c
        or "post operative" in c
        or "whipple" in c
        or "hepaticojejunostomy" in c
        or "gastrojejunostomy" in c
        or "bowel resection" in c
        or "anastomosis" in c
        or "ileostomy" in c
        or "esophagectomy" in c
    ):
        return "M_POSTSURGICAL_UPPER_GI"
    if (
        "air-fluid" in c
        or "fluid and gas" in c
        or "fluid filled" in c
        or "fluid-filled" in c
        or "bowel" in c
        or "colon" in c
        or "small bowel" in c
        or "diarrheal state" in c
        or "distended proximal" in c
        or "gas-filled loops" in c
    ):
        return "M_BOWEL_DISTENSION_FLUID"
    if "atelect" in c:
        return "M_ATELECTASIS"
    if (
        "ground-glass" in c
        or "ground glass" in c
        or "consolidat" in c
        or "pneumonic" in c
        or "infiltrat" in c
        or "air bronchogram" in c
    ):
        if (
            "ground" in c
            or "widespread" in c
            or "diffuse" in c
            or "bilateral" in c
            or "both lungs" in c
            or "multilobar" in c
        ):
            return "M_GGO_CONSOLIDATION_DIFFUSE"
        return "M_LOBAR_CONSOLIDATION"
    if (
        "atheroscler" in c
        or "atheromat" in c
        or "calcific" in c
        or "calcified" in c
        or "calcification" in c
        or "plaque" in c
        or "ectasia" in c
        or "ectatic" in c
        or "tortuosity" in c
        or "tortuous vasculature" in c
    ):
        if "abdominal" in c or "aortoiliac" in c or "iliac" in c or "visceral branches" in c:
            return "M_AORTIC_CA_ABDOMINAL"
        if "coronary" in c or "thoracic" in c or "mediastinal vascular" in c:
            return "M_AORTIC_CA_THORACIC_COR"
        if MILD_CALC_RE.search(c):
            return "M_AORTIC_CA_MILD"
        return "M_AORTIC_CA_GENERIC"
    if (
        label == "Pulmonary fibrotic sequela"
        and "sequela" in c
        and is_pulmonary
    ):
        direct_sequela = (
            "sequelae changes" in c
            or "sequelae change" in c
            or "pleuroparenchymal" in c
            or "traction bronchiectasis" in c
            or "parenchymal distortion" in c
            or "volume loss" in c
            or ("bronchiectatic changes" in c and "atelectasis" in c)
        )
        if "nodule" in c and not direct_sequela:
            return "M_LUNG_NODULE"
        return "M_PULMONARY_SEQUELA"
    if "bronchiect" in c:
        return "M_BRONCHIECTASIS"
    if "nodule" in c or "nodular" in c:
        if is_pulmonary and not is_nonpulmonary:
            return "M_LUNG_NODULE"
        return "M_NONPULMONARY_NODULE"
    if "sequela" in c and is_pulmonary:
        return "M_PULMONARY_SEQUELA"
    return "M_UNCLASSIFIED"


def _merge_current_ctrate_top10(motifs_data: dict) -> None:
    """Backfill annotations for the current all-label CT-RATE v1/openai plot."""
    audit_path = (
        REPO / "outputs" / "v1" / "audit" /
        "label_top_concepts.linear_openai.json"
    )
    if not audit_path.exists():
        return

    with audit_path.open() as f:
        audit = json.load(f)

    ds = motifs_data.setdefault("ctrate", {})
    for label, audit_entry in audit.items():
        if label not in LABEL_TO_CATEGORY:
            continue
        auc = audit_entry.get("stats", {}).get("test_auc")
        info = ds.setdefault(label, {"auc": auc, "concepts": []})
        if auc is not None:
            info["auc"] = auc
        concepts = info.setdefault("concepts", [])
        seen = {entry["concept"] for entry in concepts}
        for item in audit_entry.get("positive", [])[:10]:
            concept = item["concept"]
            if concept in seen:
                continue
            concepts.append({
                "concept": concept,
                "motif": _infer_motif(label, concept),
            })
            seen.add(concept)


def _merge_current_inspect_top10(motifs_data: dict) -> None:
    """Backfill annotations for the current full INSPECT v1/openai plot."""
    audit_path = (
        REPO / "outputs" / "v1" / "audit" /
        "phenotype__linear_openai__concept_importance.json"
    )
    if not audit_path.exists():
        return

    with audit_path.open() as f:
        audit = json.load(f)

    ds = motifs_data.setdefault("inspect", {})
    for label, audit_entry in audit.items():
        auc = audit_entry.get("stats", {}).get("test_auc")
        info = ds.setdefault(label, {"auc": auc, "concepts": []})
        if auc is not None:
            info["auc"] = auc
        concepts = info.setdefault("concepts", [])
        seen = {entry["concept"] for entry in concepts}
        for item in audit_entry.get("positive", [])[:10]:
            concept = item["concept"]
            if concept in seen:
                continue
            concepts.append({
                "concept": concept,
                "motif": _infer_motif(label, concept),
            })
            seen.add(concept)


def main() -> None:
    with (REPO / "concept_motifs.json").open() as f:
        motifs_data = json.load(f)
    with (REPO / "citations.json").open() as f:
        valid_citations = set(json.load(f))
    _merge_current_ctrate_top10(motifs_data)
    _merge_current_inspect_top10(motifs_data)

    # Manual overrides for stragglers the motif rules miss. The release ships
    # this empty: the one entry used for the paper mapped a bank-specific
    # drain-related observation string to M_SURGICAL_DRAIN and was removed
    # because the string is report-derived. Add your own bank's stragglers here.
    HARDCODE = {}

    annotations = {}
    unmapped_pairs = []  # (label_category, motif) pairs with no decision rule
    for ds_name, ds in motifs_data.items():
        annotations[ds_name] = {}
        for lbl, info in ds.items():
            cat = _category_for_label(lbl)
            label_entry = {"auc": info["auc"], "category": cat, "concepts": []}
            for entry in info["concepts"]:
                motif = entry["motif"]
                concept = entry["concept"]
                if motif == "M_UNCLASSIFIED" and concept in HARDCODE:
                    motif = HARDCODE[concept]
                # Direct substring match → tautological BLUE
                # ("pleural effusion" in concept when label = "Pleurisy; pleural
                # effusion") catches near-matches the motif lookup may miss.
                lbl_key = lbl.lower()
                concept_l = concept.lower()
                direct = False
                if cat == "DIRECT_PLEURAL":
                    direct = "pleural" in concept_l and "effusion" in concept_l
                elif cat == "DIRECT_CORONARY_CALC":
                    direct = (
                        "coronary" in concept_l
                        and (
                            "calcif" in concept_l
                            or "atheroscler" in concept_l
                            or "atheromat" in concept_l
                        )
                    )
                elif cat == "DIRECT_SEPTAL_THICKENING":
                    direct = (
                        "interlobular septal" in concept_l
                        or "crazy paving" in concept_l
                    )
                elif cat == "DIRECT_PERIBRONCHIAL":
                    direct = (
                        "peribronchial thick" in concept_l
                        or "bronchial wall thick" in concept_l
                    )
                for token in ["pleural effusion", "pericardial effusion",
                              "atelectasis", "bronchiect",
                              "ascites", "pneumothor", "hiatal hernia",
                              "diaphragmatic hernia",
                              "atherosclerot", "atheromat",
                              "consolidat", "ground-glass", "ground glass",
                              "mosaic attenuation", "lung opacity",
                              "lung nodule", "interlobular septal thicken"]:
                    if token in lbl_key and token in concept_l:
                        direct = True; break
                if direct:
                    color, cit = "BLUE", "TAUTOLOGICAL"
                else:
                    decision = DECISIONS.get((cat, motif))
                    if decision is None:
                        unmapped_pairs.append((cat, motif, lbl, concept))
                        color, cit = "RED", ""
                    else:
                        color, cit = decision
                if color in {B, P}:
                    if not cit:
                        raise SystemExit(
                            "green/purple annotation lacks support: "
                            f"{lbl!r} × {concept!r} ({cat}, {motif})"
                        )
                    if cit not in valid_citations:
                        raise SystemExit(
                            "unknown citation key: "
                            f"{cit!r} for {lbl!r} × {concept!r}"
                        )
                label_entry["concepts"].append({
                    "concept": concept,
                    "motif": motif,
                    "color": color,
                    "citation_key": cit,
                })
            annotations[ds_name][lbl] = label_entry

    with (REPO / "concept_annotations.json").open("w") as f:
        json.dump(annotations, f, indent=2)

    # Reporting
    from collections import Counter
    color_counts = Counter()
    for ds in annotations.values():
        for lbl_entry in ds.values():
            for c in lbl_entry["concepts"]:
                color_counts[c["color"]] += 1
    total_pairs = sum(color_counts.values())
    print(f"=== color distribution across {total_pairs} concept-label pairs ===")
    for k, n in color_counts.most_common():
        print(f"  {k:7s}  {n}")
    if unmapped_pairs:
        print(f"\n=== unmapped (category, motif) pairs falling through to RED: "
              f"{len(unmapped_pairs)} ===")
        unique = Counter((cat, motif) for cat, motif, _, _ in unmapped_pairs)
        for (cat, motif), n in unique.most_common(40):
            print(f"  {n:3d}× ({cat}, {motif})")
    print("\nwrote concept_annotations.json")


if __name__ == "__main__":
    main()
